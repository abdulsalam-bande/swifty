import sys
import os
import torch
import yaml

sys.path.append('./gemma_pytorch')
from gemma_pytorch.gemma.config import get_config_for_2b, get_config_for_7b


class GemmaGoogle:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        # Assign configuration to variables
        self.variant = config['VARIANT']
        self.machine_type = config['MACHINE_TYPE']
        self.weight_dir = config['WEIGHTS_DIR']

        # Ensure that the tokenizer is present
        tokenizer_path = os.path.join(self.weight_dir, 'tokenizer.model')
        assert os.path.isfile(tokenizer_path), 'Tokenizer not found!'
        print('tokenizer_path', tokenizer_path)
        # Ensure that the checkpoint is present
        ckpt_path = os.path.join(self.weight_dir, f'gemma-{self.variant}.ckpt')
        assert os.path.isfile(ckpt_path), 'PyTorch checkpoint not found!'

        # Set up model config.
        model_config = get_config_for_2b() if "2b" in self.variant else get_config_for_7b()
        model_config.tokenizer = tokenizer_path
        model_config.quant = 'quant' in self.variant
        # Instantiate the model and load the weights.
        torch.set_default_dtype(model_config.get_dtype())
        self.device = torch.device(self.machine_type)
        from gemma_pytorch.gemma.model import GemmaForCausalLM
        self.model = GemmaForCausalLM(model_config)
        self.model.load_weights(ckpt_path)
        self.model = self.model.to(self.device).eval()
        print('Loaded the model!')

    def predict(self, question):
        # Generate sample
        return self.model.generate(question, device=self.device, output_len=500)


if __name__ == '__main__':
    # import os
    #
    # # List only files in the current directory
    # entries = os.listdir('.')
    # for entry in entries:
    #     if os.path.isfile(entry):
    #         print(entry)

    LLM = GemmaGoogle('../../weights_config.yml')
    print(LLM.predict('who are you'))
