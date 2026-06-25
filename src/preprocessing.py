"""
Data  cleaning utilities for developers salary csv.
we will import this file in train.py and  or app.py
"""


import pandas as pd
import numpy as np

# constants.
TARGET = 'ConvertedCompYearly'
LOG_TARGET = 'log_salary'
SALARY_MIN= 10000
SALARY_MAX = 500000
SELECTED_FEATURES =['Country', 'YearsCode','EdLevel','Employment', 'LanguageHaveWorkedWith',
                    # new features to add
                    "DevType", # developer role
                    'OrgSize', #company size
                    'RemoteWork', # remote/hybrid / in-person.
                    'WorkExp', 
                    'Industry',
                    'Age', # Age band( use ordinal 0-6)
                    'ICorPM', #Individual Contributer  or Personal Manager
                    'DatabaseHaveWorkedWith',#-> DatabaseCount(How many dbs known)
                    'PlatformHaveWorkedWith',
                    'ToolCountWork',]
TOP_N_COUNTRIES = 25 # use target encoding 

# features that we want to target encode
TARGET_ENC_FEATURES = ['Country', 'DevType', 'Industry']

#ordinal mapping
ED_LEVEL_ORDINAL: dict[str, int] = {
    "Others": 0,
    "Primary School": 1, 
    "High school": 2,
    "Some High": 3,
    "Associate's": 4,
    "Bachelor's": 5,
    "Master's": 6,
    "Professional": 7
    }

REMOTE_ORDINAL: dict [str, int] = {
    "In-person": 0,
    "Hybrid": 1,
    "Remote": 2,
    "Other": 1
}



#Cleaning functions
def clean_years_code(series:pd.Series) ->pd.Series:
    """
    converts YearCode to numeric
    """
    series = series.copy()
    series = pd.to_numeric(series,  errors='coerce')
    return series

def clean_work_exp(series: pd.Series) -> pd.Series:
    series = series.copy()
    series =pd.to_numeric(series, errors='coerce')
    return series
def clean_education(series:pd.Series) -> pd.Series:
    """
    standardizing EdLevel into set of clean categories
    """
    mapping= {
        "Bachelor’s degree (B.A., B.S., B.Eng., etc.)": "Bachelor's",
        "Master’s degree (M.A., M.S., M.Eng., MBA, etc.)": "Masters",
        "Some college/university study without earning a degree": "Some College",
        "Professional degree (JD, MD, Ph.D, Ed.D, etc":"Professional",
        "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.":"High school",
        "Associate degree (A.A., A.S., etc.": "Associate's",
        "Primary/elementary school": "Primary school",
        "Other (please specify)": "Other"
    }
    incomplete=  series.map(mapping).fillna("Other")
    return incomplete.map(ED_LEVEL_ORDINAL)

def clean_employment(series:pd.Series) -> pd.Series:
    """
    Cleaning employment column
    """

    def simplify(val):
        if pd.isna(val):
            return np.nan
        val = str(val)
        if 'employed' in val.lower() or 'full-time' in val.lower():
            return 'Full-time'
        elif 'Independent contractor, freelancer, or self-employed': 
            return 'Freelancer/self-employed'
        elif 'student' in val.lower():
            return 'Student'
        else:
            return 'Other'
   
    return series.apply(simplify)

def clean_age(series: pd.Series) -> pd.Series:
    """Mapping age band to ordinal integers"""
    mapping ={
        "18-24 years old": 0,
        "25-34 years old": 1,
        "35-44 years old": 2,
        "45-54 years old": 3,
        "55-64 years old": 4,
        "65 years or older": 5,
        "prefer not to say": np.nan
    }
    return series.map(mapping)

def clean_org_size(series: pd.Series) -> pd.Series:
    "Mapping org_size bands to ordinal integers"
    mapping = {
        "Just me - Iam  frelancer, sole proprieter, etc,": 0,
        "Less than 20 employees" : 1,
        "20 to 99 employees": 2,
        "100 to 499 employees": 3,
        "500 to 999 employees": 4,
        "1,000 to 4,999 employees": 5,
        "5,000 to 9,999 employees": 6,
        "10,000 to more employees": 7,
        "I don't Know": np.nan
    }
    return series.map(mapping)


def cleaned_icorpm(series: pd.Series)-> pd.Series:
    """Mapping ICorpm role to binary:1 -manager, 0 for IC"""
    def _map(val):
        if pd.isna(val):
            return np.nan
        v = str(val).lower()
        if "manager" in v or "lead" in v:
            return 1
        return 0
    return series.apply(_map)

def cleaned_remote_work(series: pd.Series)-> pd.Series:
    """Mapping remote work values to intergers"""
    mapping = {
        "Remote": "Remote",
        "Hybrid( some in-person, leans heavily to flexibility)": "Hybrid",
        "Hybrid(some remote, leans heavily to flexibility)": "Hybrid",
        "In-person": "In-person",
        "Your choice (very flexible, you can come in when you want or just as needed)": np.nan
    }
    incomplete = series.map(mapping).fillna("Other")
    return incomplete.map(REMOTE_ORDINAL)

def count_languages(series: pd.Series) -> pd.Series:
    """
    convert semicolon separated language list into a count.

    Example: ' Bash/shell(all shells);Dart;SQL' -> 3
    """
    def _count(val):
        if pd.isna(val) or val == '':
            return np.nan
        return len(str(val).split(";"))
   
    return series.apply(_count)

def group_rare_countries(series: pd.Series, top_n: int = TOP_N_COUNTRIES) -> pd.Series:
    """
        Keep only the top N most common countries. Replace all others with 'Other'.
    """
    top_countries = series.value_counts().head(top_n).index.tolist()
    return series.apply(lambda x: x if x in top_countries else "Other")


def clean_industry(series: pd.Series) -> pd.Series:
    "keep only the to industries (10). merging the rest into 'Other"
    top = series.value_counts().head(10).index.tolist()
    return series.apply(lambda x : x if x in top else 'Other').fillna('Other')

def clean_dev_type(series: pd.Series) -> pd.Series:
    "sorting the primary developer role"
    def _primary(val):
        if pd.isnull(val):
            return "Other"
        low = str(val).lower()
        if "full-stack" in low:
            return "Full-stack"
        if "back-end" in low:
            return "Back-end"
        if "front-end" in low:
            return "Front-end"
        if "data scientist" in low or "machine learning" in low or "ml" in low:
            return "Data/ML"
        if "data engineer" in low or "data analyst" in low:
            return" Data/ML"
        if "devops" in low or "cloud" in low or "site reliablity" in low:
            return "Devops/Cloud"
        if "embedded"  in low or "haedware" in low:
            return "Embedded/Hardware"
        if "mobile" in low:
            return "Mobile"
        if "security" in low:
            return "security"
        if " manager" in low or "executive" in low or "director" in low:
            return "Management"
        return "Other"
    return series.apply(_primary)


def count_items(series: pd.Series) -> pd.Series:
    "count semi-colon separated items in a column; NaN if blank"
    def _count(val):
        if pd.isnull(val) or val =="":
            return np.nan
        return len(str(val).split(";"))
    
    return series.apply(_count)

# languages that consistetntly pay above average in SO salary surveys
HIGH_PAY_LANGUAGES = {
    "Go", "Rust", "Scalar", "Elixir", "Clojure", "Kotlin", "Swift", "F#", "Erlang", "Zig", 
    "OCaml", "Haskell"
    }
def has_high_pay_language(series: pd.Series) -> pd.Series:
    """ return 1 if the respondent knows any high-paying language"""

    def _check(val):
        if pd.isnull(val):
            return 0
        langs = {lang.strip() for lang in str(val).split(";")} # this is a set
        return 1 if langs & HIGH_PAY_LANGUAGES else 0 # &- intersection operator
    return series.apply(_check)



def load_and_clean(filepath: str) -> pd.DataFrame:
    """
        Load the stack overflow survey csv and return a clean df ready for the sklearn pipeline.
 
        parameters to pass: filepath to the survey csv
 
        returns the df ie features + target.
    """
    # step 0 - loading the data
    df = pd.read_csv(filepath, low_memory=False)
    print(f"Raw shape: {df.shape} \n")
 
    #step 1 - keep only rows with a valid salary and do some filtering.
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET].between(SALARY_MIN, SALARY_MAX)]
    print(f"Shape after the salary filter: {df.shape}")

    # step 1.5
    df['log_salary'] =  np.log1p(df[TARGET])
 
    # step 2 -  features + target and check whether they exist in the df
    cols_needed = SELECTED_FEATURES + ['log_salary']
    cols_available = [c for c in cols_needed if c in df.columns]
 
    missing_cols = set(cols_needed) - set(cols_available)
    if missing_cols:
        print(f" You dont have column(s): {missing_cols} in your dataset")
 
    df = df[cols_available].copy()
    print(f"Selected {len(cols_available)} columns, expected 16 columns")
 
    # step 3 - cleaning individual columns
    if 'YearsCode' in df.columns:
        df['YearsCode'] = clean_years_code(df['YearsCode'])

    if "WorkExp" in df.columns:
        df["WorkExp"] = clean_work_exp(df["WorkExp"]) 
    if 'EdLevel' in df.columns:
        df['EdLevel'] = clean_education(df['EdLevel'])
 
    if 'Employment' in df.columns:
        df['Employment'] = clean_employment(df['Employment'])
    
    if "Age" in df.columns:
        df['Age'] = clean_age (df["Age"])

    if "OrgSize" in df.columns:
        df["OrgSize"] = clean_org_size(df["OrgSize"])

    if "ICorPM" in df.columns:
        df["ICorPM"] = cleaned_icorpm(df["ICorPM"])

    if "RemoteWork" in df.columns:
        df["RemoteWork"] = cleaned_remote_work(df["RemoteWork"])
 
    if "Industry" in df.columns:
        df["Industry"] = clean_industry(df["Industry"])

    if "DevType" in df.columns:
        df["DevType"] = clean_dev_type(df["DevType"])

    if 'LanguageHaveWorkedWith' in df.columns:
        df['has_high_pay_lang'] = has_high_pay_language(df["LanguageHaveWorkedWith"])
        df['LanguageCount'] = count_languages(df['LanguageHaveWorkedWith'])
        df = df.drop(columns=['LanguageHaveWorkedWith'])

    if 'DatabaseHaveWorkedWith' in df.columns:
        df['DatabaseCount'] = count_items(df['DatabaseHaveWorkedWith'])
        df = df.drop(columns=['DatabaseHaveWorkedWith'])

    if 'PlatformHaveWorkedWith' in df.columns:
        df['PlatformCount'] = count_items(df['PlatformHaveWorkedWith'])
        df = df.drop(columns=['PlatformHaveWorkedWith'])

    if 'ToolCountWork' in df.columns:
        df['ToolCountWork'] = pd.to_numeric(df["ToolCountWork"], errors='coerce')
       
 
    if 'Country' in df.columns:
        df['Country'] = group_rare_countries(df['Country'])


    EMPLOYMENT_KEEP = ["Full-time", "Freelancer/Self-employed"]
    #step 4 - filter employment(keeping only full-time and freelancers)
    if 'Employment' in df.columns:
        before = len(df)
        df = df[df['Employment'].isin(EMPLOYMENT_KEEP)]
        df["Employment"] = (df["Employment"] == "Full-time").astype(int)
    print(f"Employment filter: {before}-> {len(df)} rows,"
          f"we only kept Full-time and freelance")
    
    #step 5 -adding interaction features (polynomial features)
    df = add_interaction_features(df)


    # step 6 - drop rows where ALL features are NaN (edge case)
    df = df.dropna(how='all')
 
    print(f"Clean data shape: {df.shape}")
    print(f"Missing values per column: \n {df.isna().sum().to_string()} ")
 
    return df
 
def get_feature_columns(df: pd.DataFrame) -> tuple[list, list]:
    """
        returns categorical columns and numerical columns from the cleaned df.
        This does not include the target variable.
    """
    non_target = [c for c in df.columns if c != "log_salary"]
    target_enc = [c for c in TARGET_ENC_FEATURES if c in non_target]

    num_cols = df[non_target].select_dtypes(include=['number']).columns.tolist()
    
    print(f"Expecting {len(non_target)} got {len(target_enc) + len(num_cols)} columns")
    return target_enc, num_cols

def add_interaction_features(df):
        df['yearsCode_sq'] = df['YearsCode'] ** 2
        df['WorkExp_sq'] = df['WorkExp'] ** 2
        df['Exp_ratio'] = df['WorkExp']/ (df['YearsCode'] + 1)
        df['Tech_breadth'] = (df['LanguageCount'] + df['DatabaseCount'] + df['PlatformCount']).fillna(0)
    
        return df
