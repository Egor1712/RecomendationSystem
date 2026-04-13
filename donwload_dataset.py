import kagglehub
import consts


kagglehub.login()
path = kagglehub.competition_download('h-and-m-personalized-fashion-recommendations', path=consts.DATASET_DIRECTORY)

print("Path to dataset files:", path)
