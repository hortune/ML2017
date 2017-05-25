#coding = utf8
import pandas as pd

def load_data(upath, mpath, rpath):
    users = pd.read_csv(upath).sort_values('UserID')
    
    movies = []
    
    with open(mpath,'r',encoding="latin1") as fd:
        for line in fd:
            word = line.replace('\n','')
            word = word.split(',')
            movies.append({'movieid':word[0],'title':','.join(word[1:-1]),'genre':word[-1]})
    movies = pd.DataFrame(movies)        
    movies['genre'] = movies.genre.str.split('|')

    ratings = pd.read_csv(rpath)

    users.Age = users.Age.astype('category')
    users.Gender = users.Gender.astype('category')
    users.Ocuppation = users.Occupation.astype('category')
    
    ratings.UserID = ratings.UserID.astype('category')
    ratings.MovieID = ratings.MovieID.astype('category')

    return users, movies, ratings
users, movies, ratings = load_data('users.csv','movies.csv','train.csv')

from IPython import *
embed()
