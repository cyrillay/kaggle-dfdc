from fastai.vision import *
from fastai.metrics import error_rate
from fastai.vision.learner import cnn_learner
from fastai.vision import ImageList, DatasetType, load_learner

batch_size = 64
path = "/Users/cyril/Documents/software/workspace/kaggle/deepfake/processed_data/faces/"

# TODO : make sur the data loading is consistent and gives the same ordering in learn.data.classes
data = ImageDataBunch\
    .from_folder(path, valid_pct=0.2, bs=12, ds_tfms=get_transforms(), size=224)\
    .normalize(imagenet_stats)

print(f"classes = {data.classes}")
print(f"Size of training data : {len(data.train_ds)}\n"
      f"Size of testing data : {len(data.valid_ds)}")
print("Size of real samples in train data : ", len([e for e in data.train_ds[:] if str(e[1]) == "real"]))
print("Size of fake samples in train data : ", len([e for e in data.train_ds[:] if str(e[1]) == "fake"]))


np.random.seed(2)
data.show_batch(rows=3, figsize=(7,6))
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(7)
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
interp.plot_top_losses(10)

learn.export('./export.pkl')
