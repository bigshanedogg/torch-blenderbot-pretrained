import torch
from torch import nn

class UnlikelihoodLoss(nn.modules.loss._Loss):
    def __init__(self, batch_size: int, timesteps: int, vocab_size: int, ignore_index: int = -100, is_log_probs: bool = True, underflow: float = 1e-5, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(UnlikelihoodLoss, self).__init__(size_average, reduce, reduction)
        assert batch_size * timesteps < vocab_size, "'vocab_size' must be greater than 'batch_size * timesteps'"
        self.batch_size = batch_size
        self.timesteps = timesteps
        self.vocab_size = vocab_size
        self.timesteps = timesteps
        self.ignore_index = ignore_index
        self.is_log_probs = is_log_probs
        self.underflow = underflow

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        '''
        :param input: (batch_size * timesteps, vocab_size)
        :param target: (batch_size * timesteps, )
        :return:
        '''
        if self.is_log_probs: input = input.exp()
        target_size = target.size(0)
        batch_size = int(target_size // self.timesteps)

        target_expanded = target.unsqueeze(0).expand(target.size(0), target.size(0))
        target_tril = target_expanded.tril(-1)
        ignore_index_mask = target_expanded.triu().to(torch.bool) * self.ignore_index
        target_tril = target_tril + ignore_index_mask

        for i in range(0, batch_size - 1):
            anchor_idx = (i + 1) * self.timesteps
            target_tril[anchor_idx:, :anchor_idx] = 0
            target_tril[:anchor_idx, anchor_idx:] = 0

        candidate_mask = target_tril.masked_fill(target_tril == target.unsqueeze(1), 0)
        negatvie_candidates = torch.zeros_like(input).scatter_(1, candidate_mask, 1)
        # exp: log_probabilities to probabilities
        # clamp: prevent underflow
        probs_not_to_be = torch.clamp(1 - input, min=self.underflow)
        # convert probabilities to log_probailities again
        loss = -1 * torch.log(probs_not_to_be) * negatvie_candidates
        loss = loss.sum()
        return loss

class UnlikelihoodLoss2d(nn.modules.loss._Loss):
    def __init__(self, timesteps: int, vocab_size: int, ignore_index: int = -100, ngram: int = 5, is_log_probs: bool = True, underflow: float = 1e-5, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(UnlikelihoodLoss2d, self).__init__(size_average, reduce, reduction)
        assert timesteps < vocab_size, "'vocab_size' must be greater than 'batch_size * timesteps'"
        self.timesteps = timesteps
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.ngram = ngram
        self.is_log_probs = is_log_probs
        self.underflow = underflow

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        '''
        :param input: (batch_size, timesteps, vocab_size)
        :param target: (batch_size, timsteps, timesteps) or (batch_size, timsteps, ngram)
        :return:
        '''
        assert len(target.shape) == 3, "Parameter 'target' must be 3D tensor: (batch_size, timesteps, timesteps)"

        # exp: log_probabilities to probabilities
        if self.is_log_probs: input = input.exp()

        loss = 0
        for input_row, target_row in zip(input, target):
            # candidate_mask = target_row
            # if not ngram_distribution:
            #     # make subsequent mask as a default
            #     target_expanded = target_row.unsqueeze(0).expand(self.timesteps, self.timesteps)
            #     target_tril = target_expanded.tril(-1)
            #     ignore_index_mask = target_expanded.triu().to(torch.bool) * self.ignore_index
            #     target_tril = target_tril + ignore_index_mask
            #     candidate_mask = target_tril.masked_fill(target_tril == target_row.unsqueeze(1), 0)

            # clamp: prevent underflow
            probs_not_to_be = torch.clamp(1 - input_row, min=self.underflow)
            negatvie_candidates = torch.zeros_like(input_row).scatter_(1, target_row, 1)
            # # convert probabilities to log_probailities back
            # _loss = -1 * torch.log(probs_not_to_be) * negatvie_candidates
            _loss = probs_not_to_be * negatvie_candidates
            _loss[:, self.ignore_index] = 0
            loss += _loss.sum()
        return loss

    def get_accuracy(self, predictions, targets):
        '''
        :param input: (batch_size, timesteps, vocab_size)
        :param target: (batch_size, timsteps, timesteps) or (batch_size, timsteps, ngram)
        :return:
        '''
        # predictions: (batch_size * timesteps, vocab_size)
        predictions = predictions.view(-1, predictions.size(-1))
        # predictions: (batch_size * timesteps, )
        predictions = torch.argmax(predictions, axis=-1)
        # targets: (batch_size * timesteps, )
        targets = targets.contiguous().view(-1)

        numerator = 0
        denominator = (targets != self.ignore_index).float().sum() + 1e-5
        for idx in range(0, len(predictions)):
            if targets[idx] == self.ignore_index: continue
            if idx % self.timesteps != 0:
                begin_idx = idx - idx % self.timesteps
                if predictions[idx] in targets[begin_idx:idx]: continue
            numerator += 1
        accuracy = numerator / denominator
        return accuracy