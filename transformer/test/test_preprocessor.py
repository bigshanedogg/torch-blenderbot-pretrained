# import pytest
# import numpy as np
#
# def test_tokenize(preprocessor_fixture):
#     test_sentences = [
#         "351번째 문제와 삼천오백구번 문제는 해결했다.",
#         "삼팔구호기에서 예닐곱개의 현상 관측해 세개의 보고서를 작성했다.",
#         "약 구십여개의 에러가 발생했는데, 그 중 삼천팔백팔번째 문제만 남았다.",
#         "세네번째 문제는 어렵지만 푸는 게 不可能하지는 않다."
#     ]
#
#     print("\n########### tokenize test ###########")
#     for test_sentence in test_sentences:
#         test_result = preprocessor_fixture.tokenize(test_sentence)
#         print(test_result)
#         assert len(test_result) > 0
#
# def test_sentence_to_ids(preprocessor_fixture):
#     test_sentences = [
#         "351번째 문제와 삼천오백구번 문제는 해결했다.",
#         "삼팔구호기에서 예닐곱개의 현상 관측해 세개의 보고서를 작성했다.",
#         "약 구십여개의 에러가 발생했는데, 그 중 삼천팔백팔번째 문제만 남았다.",
#         "세네번째 문제는 어렵지만 푸는 게 不可能하지는 않다."
#     ]
#
#     print("\n########### sentence_to_ids test ###########")
#     mask = True
#     encode_pos = False
#     for test_sentence in test_sentences:
#         test_result = preprocessor_fixture.sentence_to_ids(test_sentence, mask=mask, encode_pos=encode_pos)
#         print(test_result)
#         assert len(test_result) > 0
#
# def test_decode(preprocessor_fixture):
#     test_rows = [
#         [1, 2, 3, 4, 5],
#         [1, 2, 3, 4, 5, 6, 7, 8],
#         [1, 2, 3, 4],
#         [1, 2, 3, 4, 5, 6, 7]
#     ]
#
#     print("\n########### decode test ###########")
#     mask = False
#     encode_pos = False
#     for test_row in test_rows:
#         row = preprocessor_fixture.sentence_to_ids(test_row, mask=mask, encode_pos=encode_pos)
#         test_result = preprocessor_fixture.decode(row=row)
#         print(test_result)
#         assert len(test_result) > 0
#
# def test_filter_by_length(preprocessor_fixture):
#     test_rows = [
#         [1, 2, 3, 4, 5],
#         [1, 2, 3, 4, 5, 6, 7, 8],
#         [1, 2, 3, 4],
#         [1, 2, 3, 4, 5, 6, 7]
#     ]
#
#     print("\n########### filter_by_length test ###########")
#     timesteps = 7
#     approach = "ignore"
#     margin = 2
#
#     test_result = preprocessor_fixture.filter_by_length(rows=test_rows, timesteps=timesteps, approach=approach, margin=margin)
#     assert len(test_result) == 2
#
# def test_pad_row(preprocessor_fixture):
#     test_row = [1, 2, 3, 4, 5, 6, 7, 8]
#
#     print("\n########### pad_rows test ###########")
#     timesteps = 10
#     test_result = preprocessor_fixture.pad_row(rows=test_row, padding_value=0, post=True)
#     test_result = np.array(test_result)
#     assert len(test_result) == timesteps
#
# def test_onehot_ids(preprocessor_fixture):
#     test_row = [1, 2, 0, 0, 0, 1, 2]
#
#     print("\n########### onehot_rows test ###########")
#     num_class = 3
#     test_result = preprocessor_fixture.onehot_ids(ids=test_row, num_class=num_class)
#     test_result = np.array(test_result)
#     assert test_result.shape == (len(test_row), num_class)
#
