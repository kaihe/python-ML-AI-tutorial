from datasets import load_dataset
import pandas as pd
import pandas as pd

dataset = load_dataset('mstz/titanic')
df = pd.DataFrame(dataset['train'])

def get_passenger_count():
    # return the number of passengers in the dataset
    pass

def get_median_age():
    # return the median age of the passengers
    pass

def get_gender_count():
    # count the number of male and female passengers
    pass

def fill_nan_age_with_mean():
    # fill NaN age values with mean age
    pass

def sort_by_family_name():
    # Make family members are near eachother in the dataframe.
    # This is done by sorting the dataframe by the 'name' column.
    pass

def find_lowest_fare_pclass1():
    # Find the lowest fare for passengers in the first class cabin.
    # This is determined by finding the lowest 'Fare' value in the 'Pclass' column where Pclass equals 1.
    pass

if __name__ == '__main__':
    print(get_passenger_count())
    print(get_median_age())
    print(get_gender_count())
    print(fill_nan_age_with_mean())
    print(sort_by_family_name())
    print(find_lowest_fare_pclass1())