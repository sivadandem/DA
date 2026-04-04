# 🐼 Complete Pandas for Data Analysis — From Basics to Advanced

> A complete, beginner-to-advanced reference guide for Pandas with examples, multiple approaches, and practice problems.

---

## 📋 Table of Contents

1. [What is Pandas?](#1-what-is-pandas)
2. [Installation & Import](#2-installation--import)
3. [Core Data Structures](#3-core-data-structures)
   - [Series](#31-series)
   - [DataFrame](#32-dataframe)
4. [Creating DataFrames](#4-creating-dataframes)
5. [Reading & Writing Data](#5-reading--writing-data)
6. [Exploring Data](#6-exploring-data)
7. [Selecting & Indexing Data](#7-selecting--indexing-data)
8. [Filtering Data](#8-filtering-data)
9. [Adding & Removing Columns/Rows](#9-adding--removing-columnsrows)
10. [Sorting Data](#10-sorting-data)
11. [Renaming & Reindexing](#11-renaming--reindexing)
12. [Handling Missing Data](#12-handling-missing-data)
13. [Data Types & Type Casting](#13-data-types--type-casting)
14. [String Operations](#14-string-operations)
15. [DateTime Operations](#15-datetime-operations)
16. [GroupBy & Aggregation](#16-groupby--aggregation)
17. [Merging, Joining & Concatenating](#17-merging-joining--concatenating)
18. [Pivot Tables & Cross Tabs](#18-pivot-tables--cross-tabs)
19. [Apply, Map & Lambda Functions](#19-apply-map--lambda-functions)
20. [Window Functions](#20-window-functions)
21. [Multi-Level Index (MultiIndex)](#21-multi-level-index-multiindex)
22. [Performance Optimization](#22-performance-optimization)
23. [Visualization with Pandas](#23-visualization-with-pandas)
24. [Practice Problems](#24-practice-problems)

---

## 1. What is Pandas?

**Pandas** is an open-source Python library for **data manipulation and analysis**. It provides two primary data structures:

- **Series** — 1-dimensional labeled array
- **DataFrame** — 2-dimensional labeled table (like Excel/SQL)

Pandas is built on top of **NumPy** and is the backbone of almost every Data Analysis/Data Science workflow in Python.

```
Raw Data  →  Pandas (Load, Clean, Transform, Analyze)  →  Insights / Visualization
```

---

## 2. Installation & Import

### Installation

```bash
pip install pandas
pip install pandas numpy  # recommended with numpy
```

### Import

```python
import pandas as pd          # standard alias
import numpy as np           # often used alongside pandas

# Check version
print(pd.__version__)
```

---

## 3. Core Data Structures

### 3.1 Series

A **Series** is a one-dimensional labeled array that can hold any data type (integers, strings, floats, objects, etc.).

**Structure:**
```
Index  |  Value
-------+-------
  0    |   10
  1    |   20
  2    |   30
```

#### Creating a Series

```python
# --- Approach 1: From a list ---
s = pd.Series([10, 20, 30, 40])
print(s)
# 0    10
# 1    20
# 2    30
# 3    40
# dtype: int64

# --- Approach 2: From a list with custom index ---
s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
print(s)
# a    10
# b    20
# c    30

# --- Approach 3: From a dictionary ---
s = pd.Series({'alice': 90, 'bob': 85, 'charlie': 92})
print(s)
# alice      90
# bob        85
# charlie    92

# --- Approach 4: From a scalar (broadcast) ---
s = pd.Series(5, index=[0, 1, 2, 3])
print(s)
# 0    5
# 1    5
# 2    5
# 3    5

# --- Approach 5: From a NumPy array ---
s = pd.Series(np.array([100, 200, 300]))
```

#### Series Attributes

```python
s = pd.Series([10, 20, 30], index=['a', 'b', 'c'], name='scores')

s.values       # array([10, 20, 30])         → underlying NumPy array
s.index        # Index(['a', 'b', 'c'])       → index labels
s.dtype        # dtype('int64')               → data type
s.name         # 'scores'                     → name of series
s.shape        # (3,)                         → shape
s.size         # 3                            → number of elements
s.ndim         # 1                            → number of dimensions
```

#### Accessing Elements in Series

```python
s = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])

# By label
s['a']         # 10
s[['a', 'c']]  # returns a Series with a and c

# By position
s.iloc[0]      # 10
s.iloc[0:2]    # first two elements

# Boolean mask
s[s > 20]      # returns elements > 20
```

---

### 3.2 DataFrame

A **DataFrame** is a 2-dimensional labeled data structure with columns of potentially different types. Think of it as a spreadsheet or SQL table.

**Structure:**
```
       name    age    score
  0    Alice   25     90
  1    Bob     30     85
  2    Charlie 22     92
```

#### DataFrame Attributes

```python
df = pd.DataFrame({'name': ['Alice', 'Bob'], 'age': [25, 30], 'score': [90, 85]})

df.shape        # (2, 3)                → (rows, columns)
df.columns      # Index(['name', 'age', 'score'])
df.index        # RangeIndex(start=0, stop=2, step=1)
df.dtypes       # data types of each column
df.size         # 6 → total number of elements
df.ndim         # 2
df.values       # underlying NumPy array
```

---

## 4. Creating DataFrames

```python
# --- Approach 1: From a dictionary of lists ---
df = pd.DataFrame({
    'name':  ['Alice', 'Bob', 'Charlie'],
    'age':   [25, 30, 22],
    'score': [90, 85, 92]
})

# --- Approach 2: From a list of dictionaries ---
data = [
    {'name': 'Alice',   'age': 25, 'score': 90},
    {'name': 'Bob',     'age': 30, 'score': 85},
    {'name': 'Charlie', 'age': 22, 'score': 92}
]
df = pd.DataFrame(data)

# --- Approach 3: From a list of lists with column names ---
df = pd.DataFrame(
    [['Alice', 25, 90], ['Bob', 30, 85]],
    columns=['name', 'age', 'score']
)

# --- Approach 4: From a NumPy array ---
arr = np.array([[1, 2, 3], [4, 5, 6]])
df = pd.DataFrame(arr, columns=['A', 'B', 'C'])

# --- Approach 5: From a Series ---
s1 = pd.Series([1, 2, 3], name='A')
s2 = pd.Series([4, 5, 6], name='B')
df = pd.concat([s1, s2], axis=1)

# --- Approach 6: Empty DataFrame with defined columns ---
df = pd.DataFrame(columns=['name', 'age', 'score'])

# --- Approach 7: With custom index ---
df = pd.DataFrame(
    {'score': [90, 85, 92]},
    index=['alice', 'bob', 'charlie']
)
```

---

## 5. Reading & Writing Data

### Reading Data

```python
# --- CSV ---
df = pd.read_csv('file.csv')
df = pd.read_csv('file.csv', sep=';')              # different separator
df = pd.read_csv('file.csv', header=None)          # no header row
df = pd.read_csv('file.csv', names=['a','b','c'])  # assign column names
df = pd.read_csv('file.csv', index_col='id')       # set column as index
df = pd.read_csv('file.csv', usecols=['a', 'b'])   # read only specific columns
df = pd.read_csv('file.csv', nrows=100)            # read first 100 rows
df = pd.read_csv('file.csv', skiprows=5)           # skip first 5 rows
df = pd.read_csv('file.csv', na_values=['NA','?']) # define null values
df = pd.read_csv('file.csv', dtype={'age': int})   # specify column dtypes

# --- Excel ---
df = pd.read_excel('file.xlsx')
df = pd.read_excel('file.xlsx', sheet_name='Sheet1')
df = pd.read_excel('file.xlsx', sheet_name=0)

# --- JSON ---
df = pd.read_json('file.json')

# --- SQL ---
import sqlite3
conn = sqlite3.connect('database.db')
df = pd.read_sql('SELECT * FROM table', conn)
df = pd.read_sql_query('SELECT * FROM users WHERE age > 25', conn)

# --- From URL ---
df = pd.read_csv('https://example.com/data.csv')

# --- Clipboard (copy from Excel/browser) ---
df = pd.read_clipboard()
```

### Writing Data

```python
# --- CSV ---
df.to_csv('output.csv', index=False)          # index=False avoids saving row numbers
df.to_csv('output.csv', sep='\t')             # tab separated

# --- Excel ---
df.to_excel('output.xlsx', index=False)
df.to_excel('output.xlsx', sheet_name='Data', index=False)

# Multiple sheets
with pd.ExcelWriter('output.xlsx') as writer:
    df1.to_excel(writer, sheet_name='Sheet1')
    df2.to_excel(writer, sheet_name='Sheet2')

# --- JSON ---
df.to_json('output.json')
df.to_json('output.json', orient='records')   # list of records

# --- SQL ---
df.to_sql('table_name', conn, if_exists='replace', index=False)
```

---

## 6. Exploring Data

```python
df = pd.read_csv('data.csv')

# Basic info
df.head()            # first 5 rows
df.head(10)          # first 10 rows
df.tail()            # last 5 rows
df.tail(3)           # last 3 rows
df.shape             # (rows, columns)
df.info()            # column names, dtypes, non-null counts, memory usage
df.describe()        # statistical summary for numeric columns
df.describe(include='all')     # includes object/string columns too
df.describe(include='object')  # only string columns

# Column info
df.columns           # list of column names
df.dtypes            # dtype of each column
df.index             # row index

# Counts
df.count()           # non-null count per column
df.nunique()         # number of unique values per column
df['col'].value_counts()         # frequency of each unique value
df['col'].value_counts(normalize=True)  # as proportions (0 to 1)

# Memory
df.memory_usage(deep=True)       # memory usage per column in bytes

# Sample
df.sample(5)         # 5 random rows
df.sample(frac=0.1)  # random 10% of data
```

---

## 7. Selecting & Indexing Data

### Selecting Columns

```python
# --- Single column → returns Series ---
df['name']
df.name          # dot notation (avoid if column name has spaces)

# --- Multiple columns → returns DataFrame ---
df[['name', 'age']]

# --- All columns except some ---
df.drop(columns=['score'])
df[df.columns.difference(['score'])]
```

### Selecting Rows

```python
# --- By label: .loc[row_label, col_label] ---
df.loc[0]              # row with index label 0
df.loc[0:2]            # rows 0 to 2 (INCLUSIVE in loc)
df.loc[0, 'name']      # single value: row 0, column 'name'
df.loc[0:2, 'name':'age']  # rows 0-2, columns name to age

# --- By position: .iloc[row_pos, col_pos] ---
df.iloc[0]             # first row (position 0)
df.iloc[0:2]           # rows 0 to 1 (EXCLUSIVE end in iloc)
df.iloc[0, 1]          # row 0, column 1
df.iloc[0:3, 0:2]      # rows 0-2, columns 0-1
df.iloc[-1]            # last row
df.iloc[:, 0]          # all rows, first column

# --- at / iat: fast scalar access ---
df.at[0, 'name']       # label-based, single value (faster than loc for single cell)
df.iat[0, 1]           # position-based, single value (faster than iloc for single cell)
```

### .loc vs .iloc — Key Difference

```python
df = pd.DataFrame({'A': [10, 20, 30]}, index=[5, 10, 15])

df.loc[5]     # row with INDEX LABEL 5  → 10  (uses label)
df.iloc[0]    # row at POSITION 0       → 10  (uses position)

df.loc[5:10]  # rows with labels 5 to 10  → INCLUSIVE (2 rows)
df.iloc[0:2]  # rows at positions 0 to 2  → EXCLUSIVE end (2 rows)
```

---

## 8. Filtering Data

### Basic Filtering (Boolean Indexing)

```python
# Single condition
df[df['age'] > 25]
df[df['name'] == 'Alice']
df[df['score'] != 90]

# --- Approach 1: Bracket notation ---
df[df['age'] > 25]

# --- Approach 2: .query() string ---
df.query('age > 25')
df.query('age > 25 and score >= 90')
df.query('name == "Alice"')

# Using a variable in query
min_age = 25
df.query('age > @min_age')
```

### Multiple Conditions

```python
# AND condition
df[(df['age'] > 25) & (df['score'] >= 90)]
df.query('age > 25 & score >= 90')

# OR condition
df[(df['age'] > 25) | (df['score'] >= 90)]
df.query('age > 25 | score >= 90')

# NOT condition
df[~(df['age'] > 25)]
df.query('not age > 25')
```

### isin / between / str filters

```python
# isin — check if value is in a list
df[df['name'].isin(['Alice', 'Charlie'])]
df[~df['name'].isin(['Bob'])]           # NOT in list

# between — inclusive range check
df[df['age'].between(20, 30)]           # 20 <= age <= 30

# String filters
df[df['name'].str.startswith('A')]
df[df['name'].str.endswith('e')]
df[df['name'].str.contains('lic')]      # contains substring
df[df['name'].str.contains('al', case=False)]  # case-insensitive

# isnull / notnull
df[df['score'].isnull()]                # rows where score is NaN
df[df['score'].notnull()]               # rows where score is not NaN
```

---

## 9. Adding & Removing Columns/Rows

### Adding Columns

```python
# --- Approach 1: Direct assignment ---
df['grade'] = 'A'                        # constant value
df['age_plus_10'] = df['age'] + 10       # calculated column

# --- Approach 2: insert() — adds at specific position ---
df.insert(1, 'grade', 'A')               # insert at column position 1

# --- Approach 3: assign() — method chaining friendly ---
df = df.assign(grade='A', age_plus_10=df['age'] + 10)

# --- Approach 4: Using apply ---
df['category'] = df['score'].apply(lambda x: 'Pass' if x >= 60 else 'Fail')
```

### Adding Rows

```python
# --- Approach 1: pd.concat() — recommended ---
new_row = pd.DataFrame([{'name': 'Dave', 'age': 28, 'score': 88}])
df = pd.concat([df, new_row], ignore_index=True)

# --- Approach 2: loc with new index ---
df.loc[len(df)] = ['Dave', 28, 88]

# Note: df.append() was deprecated in Pandas 2.0, use pd.concat() instead
```

### Removing Columns

```python
# --- Approach 1: drop() ---
df.drop(columns=['grade'])               # returns new df
df.drop(columns=['grade', 'age_plus_10'])
df.drop('grade', axis=1)                 # axis=1 means column

# --- Approach 2: inplace=True ---
df.drop(columns=['grade'], inplace=True) # modifies original

# --- Approach 3: del ---
del df['grade']                          # modifies original directly

# --- Approach 4: pop() ---
removed_col = df.pop('grade')            # removes and returns the column as Series
```

### Removing Rows

```python
# By index label
df.drop(0)                   # drop row with label 0
df.drop([0, 1, 2])           # drop multiple rows
df.drop(index=0)             # explicit keyword

# Drop duplicates
df.drop_duplicates()
df.drop_duplicates(subset=['name'])        # based on specific columns
df.drop_duplicates(keep='last')            # keep last occurrence
```

---

## 10. Sorting Data

```python
# Sort by a single column (ascending by default)
df.sort_values('age')
df.sort_values('age', ascending=False)   # descending

# Sort by multiple columns
df.sort_values(['age', 'score'])                          # both ascending
df.sort_values(['age', 'score'], ascending=[True, False]) # mixed

# Sort by index
df.sort_index()
df.sort_index(ascending=False)

# inplace sorting
df.sort_values('age', inplace=True)

# Sort and reset index
df.sort_values('age').reset_index(drop=True)

# nlargest / nsmallest — get top N rows
df.nlargest(3, 'score')          # top 3 by score
df.nsmallest(3, 'age')           # bottom 3 by age
df.nlargest(3, ['score', 'age']) # break ties with age
```

---

## 11. Renaming & Reindexing

### Renaming Columns

```python
# --- Approach 1: rename() with dict ---
df.rename(columns={'name': 'full_name', 'age': 'years'})

# --- Approach 2: rename() with a function ---
df.rename(columns=str.upper)           # all columns to uppercase
df.rename(columns=str.lower)           # all columns to lowercase
df.rename(columns=lambda x: x.strip()) # strip whitespace

# --- Approach 3: Reassign df.columns ---
df.columns = ['full_name', 'years', 'score']

# --- Approach 4: add_prefix / add_suffix ---
df.add_prefix('col_')                  # col_name, col_age, col_score
df.add_suffix('_2024')
```

### Setting & Resetting Index

```python
# Set a column as the index
df.set_index('name')
df.set_index('name', inplace=True)

# Reset index back to default 0,1,2...
df.reset_index()            # old index becomes a column
df.reset_index(drop=True)   # old index is dropped entirely
```

---

## 12. Handling Missing Data

### Detecting Missing Data

```python
df.isnull()                  # boolean DataFrame — True where NaN
df.notnull()                 # boolean DataFrame — True where NOT NaN
df.isnull().sum()            # count of NaN per column
df.isnull().sum().sum()      # total NaN count in entire DataFrame
df.isnull().any()            # True for columns that have at least one NaN
df.isnull().all()            # True for columns where ALL values are NaN
df.isnull().mean()           # percentage of NaN per column (as fraction)
```

### Dropping Missing Data

```python
df.dropna()                  # drop rows with ANY NaN
df.dropna(how='all')         # drop rows where ALL values are NaN
df.dropna(subset=['age'])    # drop rows where 'age' is NaN
df.dropna(axis=1)            # drop columns with ANY NaN
df.dropna(thresh=2)          # keep rows with at least 2 non-NaN values
```

### Filling Missing Data

```python
# --- Approach 1: fillna() with a constant ---
df.fillna(0)
df['age'].fillna(0)
df.fillna({'age': 0, 'score': 50})  # different fill per column

# --- Approach 2: fillna() with statistics ---
df['age'].fillna(df['age'].mean())
df['age'].fillna(df['age'].median())
df['name'].fillna(df['name'].mode()[0])  # most frequent value

# --- Approach 3: Forward fill (propagate last valid value) ---
df.fillna(method='ffill')   # fills NaN with previous row's value
df.ffill()                  # shorthand (Pandas 2.0+)

# --- Approach 4: Backward fill ---
df.fillna(method='bfill')
df.bfill()                  # shorthand

# --- Approach 5: Interpolate ---
df['score'].interpolate()             # linear interpolation
df['score'].interpolate(method='polynomial', order=2)

# --- Approach 6: Replace ---
df.replace(np.nan, 0)
df.replace({'age': np.nan}, 0)
```

---

## 13. Data Types & Type Casting

```python
# Check dtypes
df.dtypes

# Convert column types
df['age'] = df['age'].astype(int)
df['score'] = df['score'].astype(float)
df['name'] = df['name'].astype(str)
df['active'] = df['active'].astype(bool)

# Convert to category (memory efficient for repeated strings)
df['gender'] = df['gender'].astype('category')

# Convert multiple columns at once
df = df.astype({'age': int, 'score': float})

# Convert to numeric (handles errors)
df['age'] = pd.to_numeric(df['age'])
df['age'] = pd.to_numeric(df['age'], errors='coerce')   # invalid → NaN
df['age'] = pd.to_numeric(df['age'], errors='ignore')   # invalid → unchanged

# Convert to datetime
df['date'] = pd.to_datetime(df['date'])
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

# Check if numeric
df['age'].dtype.kind in 'biufc'   # True for numeric types

# Memory-efficient integer types
df['age'] = df['age'].astype('int32')  # instead of int64
df['score'] = df['score'].astype('float32')
```

---

## 14. String Operations

All string operations live under the `.str` accessor.

```python
s = pd.Series(['  Alice ', 'bob', 'CHARLIE', 'dave123'])

# Case
s.str.upper()             # 'ALICE', 'BOB', 'CHARLIE', 'DAVE123'
s.str.lower()             # '  alice ', 'bob', 'charlie', 'dave123'
s.str.title()             # '  Alice ', 'Bob', 'Charlie', 'Dave123'
s.str.capitalize()        # first letter capital

# Whitespace
s.str.strip()             # remove leading/trailing whitespace
s.str.lstrip()            # left strip
s.str.rstrip()            # right strip

# Substring & Search
s.str.contains('ali', case=False)     # boolean: contains substring
s.str.startswith('A')                 # boolean
s.str.endswith('e')                   # boolean
s.str.find('li')                      # position of first occurrence

# Replace
s.str.replace('alice', 'Alice')
s.str.replace(r'\d+', '', regex=True)   # remove digits using regex

# Split
s.str.split(',')                       # split into list
s.str.split(',', expand=True)          # split into DataFrame columns
s.str.split(',').str[0]                # get first element after split

# Length
s.str.len()

# Padding
s.str.pad(10, side='left', fillchar='0')    # zero-pad
s.str.zfill(5)                              # zero-fill

# Extract with regex
s = pd.Series(['id_001', 'id_002', 'id_003'])
s.str.extract(r'(\d+)')             # extract number into a column
s.str.extractall(r'(\d+)')          # extract all matches

# Check patterns
s.str.isdigit()
s.str.isalpha()
s.str.isalnum()

# Concatenate strings
s.str.cat(sep=', ')                 # joins all into one string
```

---

## 15. DateTime Operations

```python
# --- Create datetime series ---
dates = pd.to_datetime(['2024-01-01', '2024-06-15', '2024-12-31'])
df['date'] = pd.to_datetime(df['date'])

# --- Access components via .dt accessor ---
df['date'].dt.year
df['date'].dt.month
df['date'].dt.day
df['date'].dt.hour
df['date'].dt.minute
df['date'].dt.second
df['date'].dt.day_name()       # 'Monday', 'Tuesday' etc.
df['date'].dt.month_name()     # 'January', 'February' etc.
df['date'].dt.quarter          # 1, 2, 3, or 4
df['date'].dt.week             # week number of year
df['date'].dt.dayofweek        # 0=Monday, 6=Sunday
df['date'].dt.dayofyear        # day number of year (1–365)
df['date'].dt.is_month_end     # True/False
df['date'].dt.is_month_start   # True/False
df['date'].dt.is_year_end
df['date'].dt.is_leap_year

# --- Date math ---
df['date'] + pd.Timedelta(days=7)          # add 7 days
df['date'] + pd.DateOffset(months=1)      # add 1 month
df['end'] - df['start']                   # difference as Timedelta

# --- Filter by date ---
df[df['date'] > '2024-06-01']
df[df['date'].dt.year == 2024]
df[df['date'].dt.month == 6]

# --- Resample (time-series groupby) ---
df.set_index('date').resample('M').sum()    # monthly sum
df.set_index('date').resample('W').mean()   # weekly mean
df.set_index('date').resample('Q').count()  # quarterly count

# Common offset aliases: D=day, W=week, M=month end, MS=month start,
#                        Q=quarter, Y=year end, H=hour, T=minute

# --- Date ranges ---
pd.date_range('2024-01-01', '2024-12-31', freq='D')   # all days
pd.date_range('2024-01-01', periods=12, freq='MS')    # 12 month-starts
pd.bdate_range('2024-01-01', '2024-01-31')            # business days only
```

---

## 16. GroupBy & Aggregation

### Basic GroupBy

```python
# groupby returns a DataFrameGroupBy object
grouped = df.groupby('department')

# Apply aggregation
grouped['salary'].mean()             # mean salary per department
grouped['salary'].sum()              # total salary per department
grouped['salary'].count()
grouped['salary'].max()
grouped['salary'].min()
grouped['salary'].std()
grouped['salary'].median()
grouped['salary'].nunique()

# Aggregation on entire DataFrame
grouped.mean()                       # mean of all numeric columns
grouped.sum()
grouped.describe()
```

### Multiple Aggregations

```python
# --- Approach 1: agg() with list ---
df.groupby('department')['salary'].agg(['mean', 'min', 'max', 'count'])

# --- Approach 2: agg() with dict (different agg per column) ---
df.groupby('department').agg({
    'salary': ['mean', 'max'],
    'age':    'median',
    'name':   'count'
})

# --- Approach 3: Named aggregations (clean column names) ---
df.groupby('department').agg(
    avg_salary=('salary', 'mean'),
    max_salary=('salary', 'max'),
    employee_count=('name', 'count')
)
```

### GroupBy with Multiple Keys

```python
df.groupby(['department', 'gender'])['salary'].mean()
df.groupby(['department', 'gender']).agg({'salary': 'mean', 'age': 'median'})
```

### transform() — keeps original index

```python
# Returns Series with same index as original DataFrame
# Useful for adding group-level stats back to original rows
df['dept_avg_salary'] = df.groupby('department')['salary'].transform('mean')
df['salary_rank_in_dept'] = df.groupby('department')['salary'].transform('rank')
```

### filter() — filter groups

```python
# Keep only groups where mean salary > 50000
df.groupby('department').filter(lambda x: x['salary'].mean() > 50000)
```

### apply() with GroupBy

```python
# Apply a custom function to each group
def top_2(group):
    return group.nlargest(2, 'salary')

df.groupby('department').apply(top_2)
```

---

## 17. Merging, Joining & Concatenating

### pd.concat() — stack DataFrames

```python
# Stack rows (vertically) — same columns
df_combined = pd.concat([df1, df2])
df_combined = pd.concat([df1, df2], ignore_index=True)  # reset index
df_combined = pd.concat([df1, df2], keys=['2023', '2024'])  # add MultiIndex keys

# Stack columns (horizontally)
df_combined = pd.concat([df1, df2], axis=1)
```

### pd.merge() — SQL-style joins

```python
# --- INNER JOIN (only matching rows) ---
pd.merge(df1, df2, on='id')
pd.merge(df1, df2, on='id', how='inner')

# --- LEFT JOIN (all rows from left) ---
pd.merge(df1, df2, on='id', how='left')

# --- RIGHT JOIN (all rows from right) ---
pd.merge(df1, df2, on='id', how='right')

# --- OUTER JOIN (all rows from both) ---
pd.merge(df1, df2, on='id', how='outer')

# --- Merge on multiple keys ---
pd.merge(df1, df2, on=['dept_id', 'year'])

# --- Different column names in each df ---
pd.merge(df1, df2, left_on='emp_id', right_on='id')

# --- Merge on index ---
pd.merge(df1, df2, left_index=True, right_index=True)
pd.merge(df1, df2, left_on='id', right_index=True)

# --- Suffixes for overlapping column names ---
pd.merge(df1, df2, on='id', suffixes=('_left', '_right'))

# --- Indicator column: shows source of each row ---
pd.merge(df1, df2, on='id', how='outer', indicator=True)
# _merge column: 'left_only', 'right_only', 'both'
```

### DataFrame.join()

```python
# join is index-based by default
df1.join(df2)                          # left join on index
df1.join(df2, how='inner')
df1.join(df2, on='id')                 # join df1's 'id' column with df2's index
```

### Merge vs Join vs Concat Summary

| Method | Use When | Default |
|--------|----------|---------|
| `pd.concat()` | Stacking dfs with same structure | Union of columns |
| `pd.merge()` | SQL-style join on column(s) | INNER join |
| `df.join()` | Quick index-based join | LEFT join |

---

## 18. Pivot Tables & Cross Tabs

### pivot_table()

```python
# Syntax: pd.pivot_table(data, values, index, columns, aggfunc)

# Mean salary by department and gender
pd.pivot_table(df,
    values='salary',
    index='department',
    columns='gender',
    aggfunc='mean',
    fill_value=0           # fill NaN with 0
)

# Multiple aggregations
pd.pivot_table(df,
    values='salary',
    index='department',
    columns='gender',
    aggfunc=['mean', 'count']
)

# Add margins (totals row/column)
pd.pivot_table(df,
    values='salary',
    index='department',
    columns='gender',
    aggfunc='mean',
    margins=True,          # adds 'All' row and column
    margins_name='Total'
)
```

### pivot() — simple reshape (no aggregation)

```python
# Requires unique index-column combinations
df.pivot(index='date', columns='category', values='value')
```

### pd.crosstab() — frequency tables

```python
# Counts occurrences
pd.crosstab(df['department'], df['gender'])

# With normalization
pd.crosstab(df['department'], df['gender'], normalize=True)
pd.crosstab(df['department'], df['gender'], normalize='index')  # row %

# With aggfunc
pd.crosstab(df['department'], df['gender'],
            values=df['salary'], aggfunc='mean')
```

### melt() — wide to long format

```python
# Convert columns into rows
df_long = df.melt(
    id_vars=['name', 'department'],    # columns to keep as-is
    value_vars=['q1_sales', 'q2_sales', 'q3_sales'],  # columns to melt
    var_name='quarter',                # name for the new variable column
    value_name='sales'                 # name for the new value column
)
```

### stack() and unstack()

```python
# stack: columns → rows (wide to long)
df.stack()

# unstack: rows → columns (long to wide)
df.unstack()
df.unstack(level=0)    # specify which level to unstack
```

---

## 19. Apply, Map & Lambda Functions

### apply() — column or row-level

```python
# On a column (element-wise is slower; use vectorized ops when possible)
df['score'].apply(lambda x: x * 2)
df['score'].apply(lambda x: 'Pass' if x >= 60 else 'Fail')

# Custom function
def grade(score):
    if score >= 90: return 'A'
    elif score >= 80: return 'B'
    elif score >= 70: return 'C'
    else: return 'F'

df['grade'] = df['score'].apply(grade)

# On entire row (axis=1)
df.apply(lambda row: row['score'] * 2 if row['age'] > 25 else row['score'], axis=1)

# On entire column (axis=0, default)
df.apply(lambda col: col.max() - col.min())   # range of each column
```

### map() — element-wise on Series

```python
# --- Approach 1: with a dict (replace values) ---
df['gender'] = df['gender'].map({'M': 'Male', 'F': 'Female'})

# --- Approach 2: with a function ---
df['name'] = df['name'].map(str.upper)

# --- Approach 3: with lambda ---
df['score'] = df['score'].map(lambda x: round(x, 2))

# Note: map() returns NaN for values not found in dict
# Use fillna() to handle those cases
```

### applymap() / map() — element-wise on DataFrame

```python
# In Pandas 2.0+, use DataFrame.map() (applymap was deprecated)
df.map(lambda x: round(x, 2) if isinstance(x, float) else x)
```

### vectorize vs apply — performance note

```python
# Vectorized operations are MUCH faster than apply
# Prefer this:
df['double_score'] = df['score'] * 2

# Over this (slower):
df['double_score'] = df['score'].apply(lambda x: x * 2)

# np.vectorize for complex logic
import numpy as np
vfunc = np.vectorize(lambda x: 'Pass' if x >= 60 else 'Fail')
df['result'] = vfunc(df['score'])
```

---

## 20. Window Functions

### Rolling — moving window

```python
# Rolling mean (moving average)
df['rolling_avg'] = df['sales'].rolling(window=7).mean()   # 7-day MA
df['rolling_sum'] = df['sales'].rolling(window=30).sum()
df['rolling_max'] = df['sales'].rolling(window=7).max()
df['rolling_std'] = df['sales'].rolling(window=7).std()

# Min periods (how many values required before computing)
df['sales'].rolling(window=7, min_periods=1).mean()   # compute even with less than 7
```

### Expanding — cumulative window

```python
# Expanding window grows from beginning to current row
df['cumulative_sum'] = df['sales'].expanding().sum()
df['cumulative_mean'] = df['sales'].expanding().mean()
df['cumulative_max'] = df['sales'].expanding().max()

# Equivalent to cumsum/cumprod
df['sales'].cumsum()
df['sales'].cumprod()
df['sales'].cummax()
df['sales'].cummin()
```

### Exponential Weighted Moving Average (EWMA)

```python
df['ewm_avg'] = df['sales'].ewm(span=7).mean()   # weight recent values more
df['ewm_avg'] = df['sales'].ewm(alpha=0.3).mean()  # alpha: smoothing factor 0-1
```

---

## 21. Multi-Level Index (MultiIndex)

### Creating a MultiIndex DataFrame

```python
# From groupby
df_multi = df.groupby(['department', 'gender']).agg({'salary': 'mean'})

# Using pd.MultiIndex
arrays = [['Q1', 'Q1', 'Q2', 'Q2'], ['North', 'South', 'North', 'South']]
index = pd.MultiIndex.from_arrays(arrays, names=['quarter', 'region'])
df_multi = pd.DataFrame({'sales': [100, 200, 150, 250]}, index=index)
```

### Accessing MultiIndex Data

```python
# .loc with tuples
df_multi.loc['Q1']             # all rows where first level is 'Q1'
df_multi.loc[('Q1', 'North')]  # specific combination

# xs() — cross section
df_multi.xs('Q1', level='quarter')
df_multi.xs('North', level='region')

# Slicing with IndexSlice
idx = pd.IndexSlice
df_multi.loc[idx['Q1', 'North'], :]
```

### Flattening MultiIndex

```python
# Reset MultiIndex to columns
df_multi.reset_index()

# Flatten column MultiIndex
df.columns = ['_'.join(col) for col in df.columns]
```

---

## 22. Performance Optimization

```python
# 1. Use categorical dtype for low-cardinality string columns
df['gender'] = df['gender'].astype('category')

# 2. Use smaller numeric types
df['age'] = df['age'].astype('int8')      # if values fit in -128 to 127
df['score'] = df['score'].astype('float32')

# 3. Use vectorized operations instead of loops or apply
df['doubled'] = df['score'] * 2          # vectorized
# NOT: df['doubled'] = df['score'].apply(lambda x: x * 2)

# 4. Use query() for filtering (faster on large DataFrames)
df.query('age > 25 and score > 80')

# 5. Read only needed columns
df = pd.read_csv('data.csv', usecols=['name', 'age'])

# 6. Read in chunks for huge files
chunk_iter = pd.read_csv('big_data.csv', chunksize=10000)
result = pd.concat([process(chunk) for chunk in chunk_iter])

# 7. Use eval() for arithmetic expressions
df.eval('result = score * 2 + age', inplace=True)

# 8. Avoid iterrows() — very slow
# BAD:
for index, row in df.iterrows():
    df.at[index, 'score'] *= 2

# GOOD: use vectorized
df['score'] *= 2

# If you must iterate, use itertuples() (faster than iterrows)
for row in df.itertuples():
    print(row.name, row.score)
```

---

## 23. Visualization with Pandas

Pandas has built-in `.plot()` based on **Matplotlib**.

```python
import matplotlib.pyplot as plt

# Line plot
df['sales'].plot()
df.plot(x='date', y='sales')

# Bar chart
df['department'].value_counts().plot(kind='bar')
df.plot(kind='bar', x='department', y='salary')
df.plot(kind='barh')                      # horizontal bar

# Histogram
df['age'].plot(kind='hist', bins=20)
df['age'].hist(bins=20)

# Box plot
df.plot(kind='box')
df['salary'].plot(kind='box')

# Scatter plot
df.plot(kind='scatter', x='age', y='salary')
df.plot.scatter(x='age', y='salary', c='score', colormap='viridis')

# Pie chart
df['department'].value_counts().plot(kind='pie', autopct='%1.1f%%')

# Area plot
df.plot(kind='area')

# Hexbin (2D density)
df.plot.hexbin(x='age', y='salary', gridsize=20)

# Styling
df['sales'].plot(
    figsize=(12, 6),
    title='Monthly Sales',
    xlabel='Month',
    ylabel='Sales ($)',
    color='steelblue',
    linewidth=2,
    grid=True
)
plt.tight_layout()
plt.savefig('plot.png', dpi=150)
plt.show()
```

---

## 24. Practice Problems

### 🟢 Beginner

**Problem 1:** Create a DataFrame with 5 employees: name, department, salary, years_experience. Then:
- Display only the name and salary columns
- Filter employees earning more than 60,000
- Sort by salary in descending order

```python
# Starter
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'Dave', 'Eve'],
    'department': ['HR', 'IT', 'IT', 'Finance', 'HR'],
    'salary': [55000, 80000, 72000, 65000, 58000],
    'years_experience': [3, 7, 5, 4, 2]
})

# Your solution here...
```

**Problem 2:** Load any CSV file and:
- Display shape, dtypes, and first 5 rows
- Count missing values per column
- Drop rows with any missing value
- Print the shape again to confirm

---

### 🟡 Intermediate

**Problem 3:** Using a sales DataFrame with columns [date, product, region, sales]:
- Convert `date` to datetime
- Extract the month and year into separate columns
- Find total sales per month
- Find the top 3 products by total sales

**Problem 4:** Merge two DataFrames:
- `employees`: [emp_id, name, dept_id]
- `departments`: [dept_id, dept_name, budget]
- Find all employees with their department name and budget
- Find employees whose department has a budget > 100,000

---

### 🔴 Advanced

**Problem 5:** Time Series Analysis:
- Create a DataFrame with daily dates for the full year 2024 and random sales values
- Compute a 7-day rolling average
- Compute month-over-month growth rate
- Identify months with above-average sales

```python
# Starter
import numpy as np
dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
df = pd.DataFrame({'date': dates, 'sales': np.random.randint(100, 1000, len(dates))})
```

**Problem 6:** Complex GroupBy:
- Given a DataFrame of students with [student_id, subject, score, semester]:
- Find the top-scoring student per subject per semester
- Find subjects where the average score dropped between semesters
- Add a column showing each student's percentile rank within their subject

**Problem 7:** Pivot and Reshape:
- Start with long-format data: [employee, quarter, metric, value]
- Pivot so each quarter is a column and each metric is a row
- Calculate QoQ (quarter-over-quarter) percentage change
- Identify the top metric by growth rate

---

## 📌 Quick Reference Cheat Sheet

```python
# ── IMPORT ─────────────────────────────────
import pandas as pd
import numpy as np

# ── CREATE ─────────────────────────────────
pd.Series([1, 2, 3])
pd.DataFrame({'a': [1,2], 'b': [3,4]})
pd.read_csv('file.csv')

# ── EXPLORE ────────────────────────────────
df.head() / df.tail()
df.shape / df.info() / df.describe()
df.columns / df.dtypes / df.index

# ── SELECT ─────────────────────────────────
df['col']             # Series
df[['col1','col2']]   # DataFrame
df.loc[0, 'col']      # by label
df.iloc[0, 1]         # by position

# ── FILTER ─────────────────────────────────
df[df['col'] > 10]
df.query('col > 10')
df[df['col'].isin(['a','b'])]
df[df['col'].between(1, 10)]

# ── CLEAN ──────────────────────────────────
df.dropna()
df.fillna(0)
df.drop_duplicates()
df['col'].astype(int)

# ── TRANSFORM ──────────────────────────────
df.sort_values('col')
df.rename(columns={'old':'new'})
df['col'].str.upper()
df['date'].dt.year

# ── AGGREGATE ──────────────────────────────
df.groupby('col').agg({'a': 'mean', 'b': 'sum'})
pd.pivot_table(df, values='v', index='i', columns='c')

# ── COMBINE ────────────────────────────────
pd.concat([df1, df2])
pd.merge(df1, df2, on='id', how='left')

# ── APPLY ──────────────────────────────────
df['col'].apply(lambda x: x*2)
df['col'].map({'a': 1, 'b': 2})

# ── SAVE ───────────────────────────────────
df.to_csv('file.csv', index=False)
df.to_excel('file.xlsx', index=False)
```

---

> 💡 **Tips for Data Analysis:**
> 1. Always explore data first with `.info()`, `.describe()`, `.head()`
> 2. Check missing values early: `df.isnull().sum()`
> 3. Check data types: `df.dtypes` — many bugs come from wrong types
> 4. Use vectorized operations over loops for speed
> 5. Chain operations with `.pipe()` for readable pipelines
> 6. Use `copy()` when modifying subsets to avoid `SettingWithCopyWarning`
> 7. Profile memory with `df.memory_usage(deep=True)` on large datasets

---

*Made for Data Analysis learners. Covers Pandas 1.x and 2.x.*