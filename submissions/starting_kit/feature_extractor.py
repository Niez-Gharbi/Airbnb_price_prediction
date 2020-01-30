from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
import datetime
import math
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from category_encoders.ordinal import OrdinalEncoder
from category_encoders.target_encoder import TargetEncoder

class FeatureExtractor(BaseEstimator,TransformerMixin):
    def __init__(self):
        self.cols_to_keep = ['city_origin', 'host_total_listings_count', 'host_since',
                        'latitude', 'amenities',
                        'longitude', 'room_type', 'accommodates', 'bathrooms', 'beds',
                        'guests_included', 'minimum_nights', 'number_of_reviews',
                        'review_scores_rating', 'cancellation_policy', 'reviews_per_month',
                        'instant_bookable', 'property_type']
        self.num_na = ['host_total_listings_count', 'bathrooms', 'beds', 'review_scores_rating', 'reviews_per_month']
        self.cat_na = ['host_since', 'property_type']
        self.amenities_to_keep = ['Well-lit path to entrance',
                                   'translation missing: en.hosting_amenity_50',
                                   'Paid parking on premises', 'No stairs or steps to enter',
                                   'Private living room', 'Self check-in', 'Pets allowed',
                                   'Free street parking', 'Buzzer/wireless intercom',
                                   'Free parking on premises', 'Extra pillows and blankets', 'Dishwasher',
                                   'Patio or balcony', 'Cable TV', 'Luggage dropoff allowed',
                                   'Smoking allowed', 'Paid parking off premises',
                                   'Carbon monoxide detector', 'Internet', 'Long term stays allowed',
                                   'Dryer', 'Microwave', 'Host greets you', 'Lock on bedroom door',
                                   'First aid kit', 'Coffee maker', 'Oven', 'Private entrance',
                                   'Family/kid friendly', 'Fire extinguisher', 'Stove', 'Bed linens',
                                   'Cooking basics', 'Elevator', 'Dishes and silverware', 'Refrigerator',
                                   'Air conditioning', 'Smoke detector', 'Iron', 'Hot water',
                                   'Laptop friendly workspace', 'Shampoo', 'TV']
        self.inmputer = SimpleImputer()

    def fit(self, X_df, y=None): 
        
        def regroup_cat(X, liste):
            if X not in liste:
                return( 'other')
            else :
                return(X)
            
        self.prop_to_keep = ['Apartment',  'Serviced apartment', 'Condominium', 'Loft']
        self.prop_transformer = TargetEncoder()
        self.prop_transformer.fit(X_df['property_type'].apply(lambda x: regroup_cat(x, self.prop_to_keep)),y) 
   
        self.pol_to_keep = ['flexible', 'strict_14_with_grace_period', 'moderate', 'moderate_new']
        self.pol_transformer = TargetEncoder()
        self.pol_transformer.fit(X_df['cancellation_policy'].apply(lambda x: regroup_cat(x, self.pol_to_keep)), y) 
               
        self.room_transformer = OrdinalEncoder()
        self.room_transformer.fit(X_df['room_type'])
        
        self.city_transformer = OneHotEncoder(handle_unknown = 'ignore')
        self.city_transformer.fit(pd.DataFrame(X_df['city_origin']))
        
       # numeric_transformer = Pipeline(steps = [('impute', SimpleImputer(strategy='median'))])
        
        return self

    def transform(self, X_df):

        def regroup_cat(X, liste):
            if X not in liste:
                return( 'other')
            else :
                return(X)
        
        def replace_all(text, dic):
            for i, j in dic.items():
                text = text.replace(i, j)
            return text
        
        X_new = X_df[self.cols_to_keep].copy()
        
        #date
        X_new['host_since'] = pd.to_datetime(X_new['host_since'], format='%Y-%m-%d').dt.year
        
        #amenities
        amenities = X_new['amenities'].apply(lambda x: replace_all(x, {'{': '', '"': '', '}': ''})).str.get_dummies(sep=',')
        X_new = pd.merge(X_new, amenities[self.amenities_to_keep], left_index=True, right_index=True)
        X_new.drop(['amenities'], axis=1, inplace=True)
        
        #fill missing
        X_new[self.num_na] = SimpleImputer().fit_transform(X_new[self.num_na])
        X_new[self.cat_na] = SimpleImputer(strategy='most_frequent').fit_transform(X_new[self.cat_na])
        
        #cat encoding
        ## concellation policy encoding
        X_new['cancellation_policy'] = self.pol_transformer.transform(X_new['cancellation_policy'].apply(lambda x: regroup_cat(x, self.pol_to_keep))) 
        
        ## proprety type
        X_new['property_type'] = self.prop_transformer.transform(X_new['property_type'].apply(lambda x: regroup_cat(x, self.prop_to_keep))) 
        
        ##room type
        X_new['room_type'] = self.room_transformer.transform(X_new['room_type'])
        
        ###city_origin_encoding
        X_new = pd.concat([X_new.reset_index(drop=True).drop(['city_origin'], axis = 1), pd.DataFrame(self.city_transformer.transform(pd.DataFrame(X_new['city_origin'])).toarray())], axis = 1)        #X_new.drop(['city_origin'], axis=1, inplace=True)
        
        #instant bookable
        X_new['instant_bookable'] = X_new['instant_bookable'].replace({"t": 1, "f": 0})
        
        
        return X_new
    

        