class LearningRateLambda:
    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)

    def transformer_lambda(self, current_step):
        initial_learning_rate = self.__dict__["initial_learning_rate"]
        num_warmup_steps = self.__dict__["num_warmup_steps"]
        arg1 = (current_step + 1) ** -0.5
        arg2 = (current_step + 1) * (num_warmup_steps ** -1.5)
        lr = initial_learning_rate * min(arg1, arg2)
        return lr