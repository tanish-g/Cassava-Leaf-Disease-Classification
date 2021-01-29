class leafEfficientNet(nn.Module):
    def __init__(self,model_name='efficientnet-b0',pool_type=F.adaptive_avg_pool2d):
        super().__init__()
        self.pool_type = pool_type
        self.backbone = EfficientNet.from_pretrained(model_name,num_classes=5)
#         in_features = getattr(self.backbone,'_fc').in_features
#         self.backbone._fc = nn.Linear(in_features,5)
    def forward(self,x):
        x=self.backbone(x)
        return x
