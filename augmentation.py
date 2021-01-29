import albumentations as A
from albumentations.pytorch.transforms import ToTensor
class get_augmentations():
  def __init__(self,img_size=256):
      self.imagenet_stats = {'mean':[0.485, 0.456, 0.406], 'std':[0.229, 0.224, 0.225]}
      self.image_size=img_size
  def train_tfms(self):
    return A.Compose([
                  A.RandomSizedCrop(min_max_height=(128,self.image_size),height=self.image_size,width=self.image_size,p=0.5),
                  A.Resize(self.image_size, self.image_size,always_apply=True,p=1.0), 
                  # A.Blur(blur_limit=7,p=0.5),
                  # A.MotionBlur(blur_limit=67,p=0.5),
                  # A.RandomFog(fog_coef_lower=0.5,fog_coef_upper=0.6),
                  #A.Transpose(p=0.5),
                  #A.VerticalFlip(p=0.5),
                  A.HorizontalFlip(p=0.5),
                  A.RandomBrightness(limit=0.2, p=0.5),
                  A.RandomContrast(limit=0.2, p=0.5),
                  A.OneOf([
                      A.MotionBlur(blur_limit=5),
                      A.MedianBlur(blur_limit=5),
                      A.GaussianBlur(blur_limit=5),
                      A.GaussNoise(var_limit=(5.0, 30.0)),
                  ], p=0.7),
                  A.OneOf([
                      A.OpticalDistortion(distort_limit=1.0),
                      A.GridDistortion(num_steps=5, distort_limit=1.),
                      A.ElasticTransform(alpha=3),
                  ], p=0.7),
                  A.ElasticTransform(alpha=3,sigma=60,alpha_affine=40,p=0.3),
                  A.CLAHE(clip_limit=4.0, p=0.7),
                  A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
                  A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.5),
                  A.Cutout(max_h_size=int(self.image_size * 0.25), max_w_size=int(self.image_size * 0.25), num_holes=1, p=0.7),
                  ToTensor(normalize=self.imagenet_stats)
                  ])
  def test_tfms(self):
      return A.Compose([
          A.Resize(self.image_size, self.image_size),                  
          ToTensor(normalize=imagenet_stats)
          ])
