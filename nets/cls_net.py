import torch.nn as nn

class Classifier_Ensemble(nn.Module):
    def __init__(self, cls_num, input_modality):
        super(Classifier_Ensemble, self).__init__()
        dim_size = 6272*2 #20608
        if input_modality == 'v':
            dim_size = 6272#512*7*7
        elif input_modality == 'a':
            dim_size = 2048
        self.fc_a = nn.Linear(6272, cls_num)
        self.fc_v = nn.Linear(6272, cls_num)

    def forward(self, feat_sound, feat_img):
        g_a = self.fc_a(feat_sound)
        g_v = self.fc_v(feat_img)
        return g_a, g_v