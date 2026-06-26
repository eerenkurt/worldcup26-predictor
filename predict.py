import pandas as pd
import joblib
import numpy as np

# 1. Model ve İşlenmiş Veriyi Yükle
try:
    model = joblib.load('football_production_model.pkl')
    df_master = pd.read_csv('final_football_data.csv')
except Exception as e:
    print(f"Hata: Model veya veri dosyaları yüklenemedi! {e}")

def get_all_teams():
    """
    Arayüzdeki açılır menüde (dropdown) listelemek için 
    veri setindeki benzersiz tüm takım isimlerini alfabetik döner.
    """
    # Hem ev sahibi hem deplasman sütunundaki benzersiz takımları birleştiriyoruz
    all_teams = sorted(list(set(df_master['home_team'].unique()) | set(df_master['away_team'].unique())))
    return all_teams

def get_team_latest_features(team_name):
    """
    Girilen takımın en güncel form ve gol istatistiklerini 
    veri setinden bulup getiren yardımcı fonksiyon.
    """
    # Takımın ev sahibi veya deplasman olarak oynadığı en son maçı buluyoruz
    latest_match = df_master[(df_master['home_team'] == team_name) | (df_master['away_team'] == team_name)].sort_values(by='date').tail(1)
    
    if latest_match.empty:
        # Eğer veri setinde takım bir şekilde bulunamazsa varsayılan nötr değerler
        return {
            'form': 0.5, 'avg_scored': 1.0, 'avg_conceded': 1.0,
            'wc_appearances': 0, 'titles': 0, 'finals': 0, 'semis': 0
        }
    
    row = latest_match.iloc[0]
    
    # Takım o son maçta ev sahibi miydi yoksa deplasman mı? 
    # Özellikleri ona göre doğru sütundan çekiyoruz.
    if row['home_team'] == team_name:
        return {
            'form': row['home_form'], 
            'avg_scored': row['home_avg_scored'], 
            'avg_conceded': row['home_avg_conceded'],
            'wc_appearances': row['home_wc_appearances'], 
            'titles': row['home_titles'], 
            'finals': row['home_finals_reached'],
            'semis': row['home_semis_reached']
        }
    else:
        return {
            'form': row['away_form'], 
            'avg_scored': row['away_avg_scored'], 
            'avg_conceded': row['away_avg_conceded'],
            'wc_appearances': row['away_wc_appearances'], 
            'titles': row['away_titles'], 
            'finals': row['away_finals_reached'],
            'semis': row['away_semis_reached']
        }

def predict_match(home_team, away_team):
    """
    Arayüzden (app.py) gelen ev sahibi ve deplasman takım isimlerine göre
    %77.30'luk CatBoost modelini çalıştırıp galibiyet olasılıklarını döner.
    """
    h_meta = get_team_latest_features(home_team)
    a_meta = get_team_latest_features(away_team)
    
    # Modelin tam olarak eğitildiği andaki 12 özelliğin (final_features) sırası:
    input_data = pd.DataFrame([{
        'home_form': h_meta['form'], 
        'away_form': a_meta['form'],
        'home_avg_scored': h_meta['avg_scored'], 
        'away_avg_scored': a_meta['avg_scored'],
        'home_avg_conceded': h_meta['avg_conceded'], 
        'away_avg_conceded': a_meta['avg_conceded'],
        'home_wc_appearances': h_meta['wc_appearances'], 
        'away_wc_appearances': a_meta['wc_appearances'],
        'home_titles': h_meta['titles'], 
        'away_titles': a_meta['titles'],
        'home_finals_reached': h_meta['finals'], 
        'away_finals_reached': a_meta['finals']
    }])
    
    # Olasılıkları hesapla (CatBoost sınıfları: [1, 2])
    # 1: Ev sahibi kazanır, 2: Deplasman kazanır (Beraberlikleri elediğimiz için 2 sınıf var)
    probs = model.predict_proba(input_data)[0]
    pred = model.predict(input_data)[0]
    
    # predict_proba çıktısındaki index sırasına göre olasılıkları eşleştiriyoruz
    return {
        'winner_code': int(pred),       # 1 veya 2
        'home_prob': float(probs[0] * 100),
        'away_prob': float(probs[1] * 100)
    }