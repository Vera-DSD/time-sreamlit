import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import io
import warnings
warnings.filterwarnings('ignore')

# Импорт моделей
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split

# Настройка страницы
st.set_page_config(
    page_title="Прогнозирование временных рядов",
    page_icon="📈",
    layout="wide"
)

# Стили CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .success-text { color: #10B981; }
    .warning-text { color: #F59E0B; }
    .error-text { color: #EF4444; }
</style>
""", unsafe_allow_html=True)

# Заголовок
st.markdown('<h1 class="main-header">📈 Прогнозирование временных рядов</h1>', unsafe_allow_html=True)
st.markdown("Загрузите файл с данными временного ряда для прогнозирования")

# Боковая панель
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103655.png", width=100)
    st.markdown("### ⚙️ Настройки")
    
    uploaded_file = st.file_uploader("📁 Загрузите CSV/Excel файл", type=['csv', 'xlsx', 'xls'])
    
    st.markdown("---")
    st.markdown("### 🎯 Параметры данных")
    
    date_column = st.text_input("Название столбца с датой", value="Order Date")
    value_column = st.text_input("Название столбца с целевой переменной", value="Sales")
    
    frequency = st.selectbox(
        "Частота данных",
        ["D (ежедневно)", "W (еженедельно)", "M (ежемесячно)", "Q (квартально)", "Y (ежегодно)"]
    )
    
    freq_map = {"D (ежедневно)": "D", "W (еженедельно)": "W", 
                "M (ежемесячно)": "M", "Q (квартально)": "Q", "Y (ежегодно)": "Y"}
    selected_freq = freq_map[frequency]
    
    forecast_periods = st.slider("Количество периодов для прогноза", 1, 52, 12)
    
    st.markdown("---")
    st.markdown("### 🤖 Модели для прогнозирования")
    
    use_arima = st.checkbox("ARIMA", value=True)
    use_prophet = st.checkbox("Prophet", value=True)
    use_rf = st.checkbox("Random Forest", value=True)
    
    if st.button("🚀 Запустить прогнозирование", type="primary", use_container_width=True):
        st.session_state.run_forecast = True

# Функции для обработки данных
def load_data(file):
    """Загрузка данных из файла"""
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        return df
    except Exception as e:
        st.error(f"Ошибка загрузки файла: {e}")
        return None

def prepare_time_series(df, date_col, value_col, freq='W'):
    """Подготовка временного ряда"""
    df = df.copy()
    
    # Конвертация даты
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Агрегация по выбранной частоте
    df['period'] = df[date_col].dt.to_period(freq).dt.start_time
    ts = df.groupby('period')[value_col].sum().reset_index()
    ts.columns = ['ds', 'y']
    
    return ts

def create_features_for_rf(series, n_lags=12):
    """Создание признаков для Random Forest"""
    df = pd.DataFrame({'y': series})
    
    # Лаги
    for lag in range(1, n_lags + 1):
        df[f'lag_{lag}'] = df['y'].shift(lag)
    
    # Скользящая статистика
    for window in [3, 6, 12]:
        df[f'rolling_mean_{window}'] = df['y'].rolling(window=window, min_periods=1).mean()
        df[f'rolling_std_{window}'] = df['y'].rolling(window=window, min_periods=1).std()
    
    # Временные признаки
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    
    return df.dropna()

# Функции моделей
def run_arima(train_data, periods):
    """Прогнозирование ARIMA"""
    try:
        model = ARIMA(train_data['y'], order=(1,1,1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=periods)
        return forecast, model_fit
    except Exception as e:
        st.warning(f"ARIMA ошибка: {e}")
        return None, None

def run_prophet(train_data, periods):
    """Прогнозирование Prophet"""
    try:
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True if len(train_data) > 7 else False)
        model.fit(train_data)
        
        future = model.make_future_dataframe(periods=periods, freq=selected_freq)
        forecast = model.predict(future)
        
        # Берем только прогнозные значения
        forecast_values = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
        return forecast_values, model
    except Exception as e:
        st.warning(f"Prophet ошибка: {e}")
        return None, None

def run_random_forest(train_data, periods):
    """Прогнозирование Random Forest"""
    try:
        # Создаем признаки
        features_df = create_features_for_rf(train_data['y'].values)
        
        if len(features_df) < 20:
            st.warning("Слишком мало данных для Random Forest")
            return None, None
        
        X = features_df.drop('y', axis=1)
        y = features_df['y']
        
        # Разделение
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Модель
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Прогноз (упрощенный)
        last_features = X.iloc[-1:].values
        forecasts = []
        
        for _ in range(periods):
            pred = model.predict(last_features)[0]
            forecasts.append(pred)
            
            # Обновляем признаки для следующего прогноза
            last_features = np.roll(last_features, 1)
            last_features[0, 0] = pred
        
        return forecasts, model
    except Exception as e:
        st.warning(f"Random Forest ошибка: {e}")
        return None, None

# Основной контент
if uploaded_file is not None:
    # Загрузка данных
    df = load_data(uploaded_file)
    
    if df is not None:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<h3 class="sub-header">📊 Предпросмотр данных</h3>', unsafe_allow_html=True)
            st.dataframe(df.head(10), use_container_width=True)
        
        with col2:
            st.markdown('<h3 class="sub-header">📋 Информация о данных</h3>', unsafe_allow_html=True)
            st.metric("Количество строк", len(df))
            st.metric("Количество столбцов", len(df.columns))
            
            if date_column in df.columns:
                date_info = pd.to_datetime(df[date_column])
                st.metric("Диапазон дат", f"{date_info.min().date()} - {date_info.max().date()}")
        
        # Подготовка временного ряда
        ts_data = prepare_time_series(df, date_column, value_column, selected_freq)
        
        # Визуализация исходных данных
        st.markdown('<h3 class="sub-header">📈 Исходный временной ряд</h3>', unsafe_allow_html=True)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ts_data['ds'],
            y=ts_data['y'],
            mode='lines+markers',
            name='Фактические данные',
            line=dict(color='#3B82F6', width=2)
        ))
        
        fig.update_layout(
            title="Динамика временного ряда",
            xaxis_title="Дата",
            yaxis_title=value_column,
            hovermode='x unified',
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Разделение на train/test
        train_size = int(len(ts_data) * 0.8)
        train_data = ts_data.iloc[:train_size]
        test_data = ts_data.iloc[train_size:]
        
        # Контейнер для результатов
        if 'run_forecast' in st.session_state and st.session_state.run_forecast:
            st.markdown("---")
            st.markdown('<h2 class="main-header">🔮 Результаты прогнозирования</h2>', unsafe_allow_html=True)
            
            # Прогресс бар
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = {}
            models = {}
            
            # Запуск выбранных моделей
            models_to_run = []
            if use_arima: models_to_run.append(('ARIMA', run_arima))
            if use_prophet: models_to_run.append(('Prophet', run_prophet))
            if use_rf: models_to_run.append(('Random Forest', run_random_forest))
            
            for i, (model_name, model_func) in enumerate(models_to_run):
                status_text.text(f"🔄 Обучение {model_name}...")
                forecast, model = model_func(train_data, forecast_periods)
                
                if forecast is not None:
                    results[model_name] = forecast
                    models[model_name] = model
                
                progress_bar.progress((i + 1) / len(models_to_run))
            
            status_text.text("✅ Прогнозирование завершено!")
            
            # Визуализация прогнозов
            if results:
                st.markdown('<h3 class="sub-header">📊 Сравнение прогнозов</h3>', unsafe_allow_html=True)
                
                fig_forecast = go.Figure()
                
                # Фактические данные
                fig_forecast.add_trace(go.Scatter(
                    x=ts_data['ds'],
                    y=ts_data['y'],
                    mode='lines',
                    name='Фактические данные',
                    line=dict(color='#3B82F6', width=2)
                ))
                
                # Прогнозы каждой модели
                colors = {'ARIMA': '#10B981', 'Prophet': '#F59E0B', 'Random Forest': '#EF4444'}
                
                for model_name, forecast in results.items():
                    if model_name == 'Prophet' and hasattr(forecast, 'columns'):
                        # Prophet возвращает DataFrame
                        fig_forecast.add_trace(go.Scatter(
                            x=forecast['ds'],
                            y=forecast['yhat'],
                            mode='lines',
                            name=f'{model_name} прогноз',
                            line=dict(color=colors.get(model_name, '#000'), width=2, dash='dash')
                        ))
                        
                        # Доверительный интервал для Prophet
                        fig_forecast.add_trace(go.Scatter(
                            x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
                            y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
                            fill='toself',
                            fillcolor='rgba(245, 158, 11, 0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name=f'{model_name} дов. интервал',
                            showlegend=True if model_name == 'Prophet' else False
                        ))
                    else:
                        # ARIMA и Random Forest
                        forecast_dates = pd.date_range(
                            start=ts_data['ds'].iloc[-1] + pd.Timedelta(days=1),
                            periods=forecast_periods,
                            freq=selected_freq
                        )
                        
                        fig_forecast.add_trace(go.Scatter(
                            x=forecast_dates,
                            y=forecast,
                            mode='lines',
                            name=f'{model_name} прогноз',
                            line=dict(color=colors.get(model_name, '#000'), width=2, dash='dash')
                        ))
                
                fig_forecast.update_layout(
                    title="Сравнение прогнозов разных моделей",
                    xaxis_title="Дата",
                    yaxis_title=value_column,
                    hovermode='x unified',
                    template='plotly_white',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig_forecast, use_container_width=True)
                
                # Метрики качества (если есть тестовые данные)
                if len(test_data) > 0:
                    st.markdown('<h3 class="sub-header">📊 Метрики качества моделей</h3>', unsafe_allow_html=True)
                    
                    metrics_data = []
                    
                    for model_name, forecast in results.items():
                        if len(forecast) >= len(test_data):
                            # Берем первые n прогнозов для сравнения с тестом
                            forecast_for_test = forecast[:len(test_data)]
                            
                            if isinstance(forecast_for_test, pd.DataFrame):
                                forecast_values = forecast_for_test['yhat'].values
                            else:
                                forecast_values = forecast_for_test
                            
                            # Рассчитываем метрики
                            mae = mean_absolute_error(test_data['y'].values, forecast_values)
                            mape = mean_absolute_percentage_error(test_data['y'].values, forecast_values) * 100
                            r2 = r2_score(test_data['y'].values, forecast_values)
                            
                            metrics_data.append({
                                'Модель': model_name,
                                'MAE': mae,
                                'MAPE (%)': mape,
                                'R²': r2,
                                'Статус': '✅ Хорошо' if mape < 20 else '⚠️ Средне' if mape < 50 else '❌ Плохо'
                            })
                    
                    if metrics_data:
                        metrics_df = pd.DataFrame(metrics_data)
                        
                        # Отображение в виде карточек
                        cols = st.columns(len(metrics_data))
                        
                        for idx, row in metrics_df.iterrows():
                            with cols[idx]:
                                st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
                                st.markdown(f"**{row['Модель']}**")
                                st.metric("MAE", f"{row['MAE']:.2f}")
                                st.metric("MAPE", f"{row['MAPE (%)']:.1f}%")
                                st.metric("R²", f"{row['R²']:.3f}")
                                st.markdown(f"**Статус:** {row['Статус']}")
                                st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Таблица с деталями
                        st.dataframe(metrics_df, use_container_width=True)
                
                # Интерпретация результатов
                st.markdown('<h3 class="sub-header">📝 Интерпретация результатов</h3>', unsafe_allow_html=True)
                
                interpretation_cols = st.columns(2)
                
                with interpretation_cols[0]:
                    st.markdown("### 🎯 Рекомендации по выбору модели")
                    
                    if 'ARIMA' in results:
                        st.markdown("""
                        **ARIMA** подходит для:
                        - Стационарных временных рядов
                        - Коротких и средних горизонтов прогнозирования
                        - Данных без сложных сезонных паттернов
                        """)
                    
                    if 'Prophet' in results:
                        st.markdown("""
                        **Prophet** подходит для:
                        - Данных с сезонностью (недельной, годовой)
                        - Учета праздников и выходных
                        - Автоматической обработки пропусков
                        """)
                    
                    if 'Random Forest' in results:
                        st.markdown("""
                        **Random Forest** подходит для:
                        - Данных с нелинейными зависимостями
                        - Больших объемов данных
                        - Когда есть дополнительные признаки
                        """)
                
                with interpretation_cols[1]:
                    st.markdown("### 💡 Практические рекомендации")
                    
                    recommendations = []
                    
                    if any('MAPE (%)' in str(m) for m in metrics_data if isinstance(m, dict)):
                        avg_mape = np.mean([m.get('MAPE (%)', 0) for m in metrics_data if isinstance(m, dict)])
                        
                        if avg_mape < 10:
                            recommendations.append("✅ **Отличная точность** - можно использовать для оперативного планирования")
                        elif avg_mape < 20:
                            recommendations.append("⚠️ **Хорошая точность** - подходит для стратегического планирования")
                        elif avg_mape < 30:
                            recommendations.append("📊 **Средняя точность** - нужна дополнительная проверка")
                        else:
                            recommendations.append("🔧 **Низкая точность** - рассмотрите другие методы или больше данных")
                    
                    recommendations.append("📈 **Собирайте больше исторических данных** для улучшения точности")
                    recommendations.append("🔄 **Регулярно переобучайте модели** на новых данных")
                    recommendations.append("🎯 **Комбинируйте прогнозы** разных моделей для лучшего результата")
                    
                    for rec in recommendations:
                        st.markdown(f"- {rec}")
                
                # Экспорт результатов
                st.markdown('<h3 class="sub-header">💾 Экспорт результатов</h3>', unsafe_allow_html=True)
                
                export_cols = st.columns(3)
                
                with export_cols[0]:
                    if st.button("📥 Скачать прогнозы CSV", use_container_width=True):
                        # Создаем DataFrame с прогнозами
                        forecast_dates = pd.date_range(
                            start=ts_data['ds'].iloc[-1] + pd.Timedelta(days=1),
                            periods=forecast_periods,
                            freq=selected_freq
                        )
                        
                        forecast_df = pd.DataFrame({'Дата': forecast_dates})
                        
                        for model_name, forecast in results.items():
                            if isinstance(forecast, pd.DataFrame):
                                forecast_df[model_name] = forecast['yhat'].values
                            else:
                                forecast_df[model_name] = forecast
                        
                        csv = forecast_df.to_csv(index=False)
                        st.download_button(
                            label="Нажмите для скачивания",
                            data=csv,
                            file_name="прогнозы_временного_ряда.csv",
                            mime="text/csv"
                        )
                
                with export_cols[1]:
                    if st.button("📊 Скачать график", use_container_width=True):
                        # Сохраняем график
                        fig_forecast.write_html("прогнозы_временного_ряда.html")
                        with open("прогнозы_временного_ряда.html", "rb") as file:
                            st.download_button(
                                label="Нажмите для скачивания",
                                data=file,
                                file_name="прогнозы_временного_ряда.html",
                                mime="text/html"
                            )
                
                with export_cols[2]:
                    if st.button("📋 Отчет в PDF", use_container_width=True):
                        st.info("Функция генерации PDF отчета в разработке")
            
            else:
                st.warning("Ни одна из моделей не смогла сделать прогноз. Проверьте данные.")
        
        else:
            st.info("Нажмите кнопку '🚀 Запустить прогнозирование' в боковой панели")
    
else:
    # Демо-режим
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📋 Формат данных")
        st.markdown("""
        Загрузите файл в одном из форматов:
        
        **CSV/Excel с колонками:**
        - Дата (например: '2024-01-01')
        - Целевая переменная (продажи, температура и т.д.)
        
        **Пример структуры:**
        """)
        
        example_df = pd.DataFrame({
            'Order Date': pd.date_range('2024-01-01', periods=10, freq='D'),
            'Sales': [100, 120, 130, 110, 140, 150, 160, 170, 180, 190]
        })
        st.dataframe(example_df, use_container_width=True)
    
    with col2:
        st.markdown("### 🚀 Возможности приложения")
        st.markdown("""
        ✅ **Поддержка нескольких моделей:**
        - ARIMA (классическая статистика)
        - Prophet (от Facebook)
        - Random Forest (машинное обучение)
        
        ✅ **Визуализация:**
        - Интерактивные графики
        - Сравнение моделей
        - Метрики качества
        
        ✅ **Экспорт результатов:**
        - CSV с прогнозами
        - HTML графики
        - Детальный отчет
        """)
        
        st.markdown("### ⚡ Быстрый старт")
        st.markdown("""
        1. Загрузите файл с данными
        2. Укажите названия колонок
        3. Выберите модели
        4. Нажмите "Запустить прогнозирование"
        """)

# Футер
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #6B7280;'>"
    "📊 Прогнозирование временных рядов | Создано с помощью Streamlit"
    "</div>",
    unsafe_allow_html=True
)