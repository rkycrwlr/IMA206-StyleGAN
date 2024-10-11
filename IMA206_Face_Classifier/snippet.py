#%%
import torch
import torch.nn as nn
from torchvision import models
from torchvision.transforms import Resize, InterpolationMode
import warnings #Pas strictement nécessaire

#%%

class EfficientNetB0(nn.Module):
    def __init__(self,img_size=512):
        super(EfficientNetB0, self).__init__()
        self.model = models.efficientnet_b0()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        if img_size == 512:
            self.head = nn.Sequential(
                nn.Linear(1280, 512),
                nn.BatchNorm1d(512),
                nn.Dropout(0.2),
                nn.Linear(512 , 128),
                nn.BatchNorm1d(128),
                nn.Dropout(0.2),
                nn.Linear(128 , 40)
                )
        elif img_size == 1024:
            self.head = nn.Sequential(
                nn.Linear(1280, 1024),
                nn.BatchNorm1d(1024),
                nn.Dropout(0.2),
                nn.Linear(1024 , 256),
                nn.BatchNorm1d(256),
                nn.Dropout(0.2),
                nn.Linear(256 , 40)
                )
        
    def forward(self,x):
        x = self.model.features(x)
        #-----#
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.head(x)
        return x
    
class Classifier(nn.Module):
    def __init__(self, attributes=torch.ones(40), from_file=None, cuda=False, sig=True):
        """
        attributes: torch.tensor
            binary tensor indicating which attribute (based on the CelebA attributes indices) to classify. 
            "1" meaning we take into account the attribute and "0" meaning we ignore it.
        from_file: str
            Path from which to load a pretrained version of the classifier.
        cuda: bool
            Boolean indicating if we must load the classifier on GPU (True) or CPU (False).
        sig: bool
            Boolean indicating if we apply a sigmoid as an activation function 
            or if we only take the logits as output.
        """
        super(Classifier, self).__init__()
        
        if from_file is None:
            warnings.warn(' "from_file" value is None: please check that you really wanna train your model from scratch. Otherwise, indicate the location of the pretrained attribute classifier.')
            self.model = EfficientNetB0()
        elif type(from_file)==str and from_file[-3:]=='.pt':
            self.model = EfficientNetB0()
            if cuda:
                self.model.load_state_dict(torch.load(from_file))
                self.model.cuda()
            else:
                self.model.load_state_dict(torch.load(from_file, map_location=torch.device('cpu')))
        else:
            raise ValueError(' "from_file" has wrong type. It is supposed to be a string (ending by ".pt") indicating the location of the pretrained attribute classifier or None.')

        self.attributes = attributes
        self.sig = sig

        self.transform = Resize(512,
                                interpolation=InterpolationMode.BICUBIC,
                                antialias=True)
        
    def forward(self, x, warn = True):
        if (x.shape[-1]!= 512 or x.shape[-2]!= 512):
            if warn:
                warnings.warn("Your input shape isn't ...x512x512, resizing has been applied to it")
            x = self.transform(x)

        x = self.model(x)
        x = x[:,torch.where(self.attributes)[0]]

        if self.sig:
            return torch.sigmoid(x)
        
        else: return x
        
    def eval(self):
        self.model.eval()
    
    def train(self,bool):
        self.model.train(bool)
        
    def requires_grad(self, requires_grad=True):
        for param in self.model.parameters():
            param.requires_grad_(requires_grad)
        self.attributes.requires_grad_(False)

# #%%

# attr = torch.zeros(40)
# #Mette à 1 les indices des attributs qu'on veut classifier (par ex, ci-dessous, on souhaite changer 'Eyeglasses')
# #Les attributs sont ceux de la base CelebA
# attr[15] = 1 

# #Créer une instance du classifier en renseignant 
# #les attrributs qu'on souhaite classifier, le chemin du modèle pré-entrainé et si on utilise le GPU ou CPU
# classifier = Classifier(attributes = attr, from_file='./classifier.pt', cuda = True)
# classifier.requires_grad(False)
# classifier.eval()

# #Petit test de classification sur une image (1024x1024) de bruit blanc:
# u = torch.randn(1,3,1024,1024).cuda()
# out = classifier(u).detach().cpu()
# print('attribute values:', out)


# %%
attr_names = ['5_o_Clock_Shadow',
        'Arched_Eyebrows',
        'Attractive',
        'Bags_Under_Eyes',
        'Bald',
        'Bangs',
        'Big_Lips',
        'Big_Nose',
        'Black_Hair',
        'Blond_Hair',
        'Blurry',
        'Brown_Hair',
        'Bushy_Eyebrows',
        'Chubby',
        'Double_Chin',
        'Eyeglasses',
        'Goatee',
        'Gray_Hair',
        'Heavy_Makeup',
        'High_Cheekbones',
        'Male',
        'Mouth_Slightly_Open',
        'Mustache',
        'Narrow_Eyes',
        'No_Beard',
        'Oval_Face',
        'Pale_Skin',
        'Pointy_Nose',
        'Receding_Hairline',
        'Rosy_Cheeks',
        'Sideburns',
        'Smiling',
        'Straight_Hair',
        'Wavy_Hair',
        'Wearing_Earrings',
        'Wearing_Hat',
        'Wearing_Lipstick',
        'Wearing_Necklace',
        'Wearing_Necktie',
        'Young']