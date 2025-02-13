from lstm import SwiftDock
model_path = '../../results_seq/serialized_models/lstm_Drp1_MiD49_clustered_2000_morgan_onehot_mac_256_model.pt'
target = 'Drp1_MiD49_clustered_2000'
llm_result = SwiftDock.inference_with_llm(model_path=model_path, smile='CCC', target=target)
print(llm_result)