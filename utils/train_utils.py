# Numerical libs
import torch

# Network wrapper, defines forward pass
class NetWrapper(torch.nn.Module):
    def __init__(self, args, nets):
        super(NetWrapper, self).__init__() 
        self.net_sound, self.net_frame, self.net_classifier = nets
        self.args = args

    def forward(self, audio, frame):
        feat_sound, feat_frame = None, None
        if self.net_sound is not None:
            feat_sound = self.net_sound(audio)
        if self.net_frame is not None:
            feat_frame = self.net_frame(frame)
        pred = self.net_classifier(feat_sound, feat_frame)
        return pred

    def get_embds(self, audio, frame):
        feat_sound, feat_frame = None, None
        if self.net_sound is not None:
            feat_sound = self.net_sound(audio)
        if self.net_frame is not None:
            feat_frame = self.net_frame(frame)
        return feat_sound, feat_frame

def adjust_learning_rate(optimizer, args):
    args.lr_sound *= 0.1
    args.lr_frame *= 0.1
    args.lr_classifier *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.1