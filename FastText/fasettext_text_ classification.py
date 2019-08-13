import fasttext as ft


# model = ft.train_supervised('/Users/shenqinb/Development/GitRepositories/PersonalGitRepositories/MachineLearning/FastText/cooking.train')
# model.save_model('/Users/shenqinb/Development/GitRepositories/PersonalGitRepositories/MachineLearning/FastText/cooking.model.bin')

model = ft.load_model('/Users/shenqinb/Development/GitRepositories/PersonalGitRepositories/MachineLearning/FastText/cooking.model.bin')
result = model.predict("Which baking dish is best to bake a banana bread ?")
print (result)

