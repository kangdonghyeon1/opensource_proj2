import pandas as pd

df = pd.read_csv('./2019_kbo_for_kaggle_v2.csv')

# 1번
print("\n1번\n ----------------------------")
df = df[(df['year'] >= 2015) & (df['year'] <= 2018)]

def top_players(df, year, column, n=10):
    df_year = df[df['year'] == year]
    return df_year.nlargest(n, column)[['batter_name', column]]

for year in range(2015, 2019):
    print(f"{year}년도 안타(H) 상위 10인")
    print(top_players(df, year, 'H'))
    print()

    print(f"{year}년도 타율(avg) 상위 10인")
    print(top_players(df, year, 'avg'))
    print()

    print(f"{year}년도 홈런(HR) 상위 10인")
    print(top_players(df, year, 'HR'))
    print()

    print(f"{year}년도 출루율(OBP) 상위 10인")
    print(top_players(df, year, 'OBP'))
    print()

# 2번
print("\n2번\n ----------------------------")
df_2018 = df[df['year'] == 2018]
idx = df_2018.groupby('cp')['war'].idxmax()

top_war_players = df_2018.loc[idx]
print(top_war_players[['batter_name', 'cp', 'war']])

# 3번
print("\n3번\n ----------------------------")
df = pd.read_csv('./2019_kbo_for_kaggle_v2.csv')

columns = ['R', 'H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG', 'salary']

df_corr = df[columns]

correlations = df_corr.corr()['salary'].drop('salary')

max_corr = correlations.idxmax()
max_corr_value = correlations.max()

print(f"연봉과 가장 높은 상관관계를 가진 변수: {max_corr}, 상관계수: {max_corr_value}")