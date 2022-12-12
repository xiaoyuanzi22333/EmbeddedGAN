import argparse
import os


class BaseOption:
    def __init__(self):
        self.initialized = False
        
        
        
    def initialize(self,parser):
        parser.add_argument('--noise_path',type=str,default='./data/noise/', required=False, help='path to load noise images')
        parser.add_argument('--encode_path',type=str,default='./data/encode/', required=False, help='path to load encode images')
        parser.add_argument('--truth_path',type=str,default='./data/truth/', required=False, help='path to load noise images')
        parser.add_argument('--model_path',type=str,default='.results/log1_pix2pix_mix', required=False, help='path to save models')
        parser.add_argument('--output_img_path',default='./results/output_pix2pix_mix',type=str, required=False, help='path to save output images')
        parser.add_argument('--writer_dir',type=str,default='.results/runs1_pix2pix_mix', required=False, help='path to save tensorboard digits')
        parser.add_argument('--num_epochs',type=int,default=50,required=False,help='total training epochs')
        parser.add_argument('--cuda_device',type=int,default=0,required=False,help='gpu_id you are going to use')
        parser.add_argument('--batch_size',type=int,default=32,required=False)
        parser.add_argument('--itr_save_img',type=int,default=50)
        parser.add_argument('--epoch_save_model',type=int,default=5)
        
        self.initialized = True
        return parser
    
    
    def get_option(self):
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

            self.parser = parser
            return parser.parse_args()
        
        

    def print_msg(self,opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)
        

    def parse(self):
        opt = self.get_option()
        self.print_msg(opt)
        self.opt = opt
        
        return self.opt
        
        