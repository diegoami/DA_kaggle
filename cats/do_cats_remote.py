import sys
sys.path.append('..')
import keras
from kaggle_imagesolver import KaggleImageSolver
if __name__ == "__main__":
    root_dir='/home/ubuntu/datasets/kaggle_cats/'

    model_dir = '/home/ubuntu/models/kaggle_cats/'
    kaggleImageSolver = KaggleImageSolver(train_dir=root_dir+'train/', test_dir=root_dir+'test/', num_categories=2)
    model = kaggleImageSolver.train_model(150, 150, model_dir=model_dir,  steps_per_epoch=10, epochs=1, batch_size=10)
    model.save(model_dir+'cats_1.hp5')

    #model = keras.models.load_model('/media/diegoami/40e5135e-5905-41f3-a006-2cd73b52e803/models/cats_1.hp5')
    #kaggleImageSolver.print_result(model,150,150,True)