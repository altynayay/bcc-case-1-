#!/usr/bin/env python3
"""
Демонстрационный скрипт для ML анализа BCC Bank
Показывает возможности комбинации классических алгоритмов и ML
"""

import pandas as pd
import numpy as np
from ml_analyzer import MLBankAnalyzer
import json
from datetime import datetime

def print_header(title):
    """Печатает красивый заголовок"""
    print("\n" + "="*60)
    print(f"🤖 {title}")
    print("="*60)

def print_section(title):
    """Печатает заголовок секции"""
    print(f"\n📊 {title}")
    print("-" * 40)

def main():
    print_header("BCC Bank ML Analysis Demo")
    print("Комбинация классических алгоритмов + Machine Learning")
    print("Для хакатона - уверенный PoC")
    
    # Инициализируем анализатор
    analyzer = MLBankAnalyzer()
    
    # Загружаем данные
    print_section("Загрузка данных")
    if not analyzer.load_data():
        print("❌ Ошибка загрузки данных")
        return
    
    print("✅ Данные успешно загружены")
    
    # Создаем признаки
    print_section("Создание инженерных признаков")
    features_df = analyzer.create_advanced_features()
    print(f"✅ Создано {len(features_df.columns)} признаков")
    print(f"📈 Примеры признаков: {list(features_df.columns[:10])}")
    
    # Кластеризация
    print_section("Кластеризация клиентов")
    clusters = analyzer.perform_clustering()
    print(f"✅ Обнаружено {len(set(clusters))} кластеров")
    
    # Обучение классификатора
    print_section("Обучение ML классификатора")
    accuracy = analyzer.train_product_classifier()
    print(f"✅ Точность модели: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    # Генерируем рекомендации
    print_section("Генерация ML рекомендаций")
    recommendations = analyzer.generate_ml_recommendations()
    print(f"✅ Создано {len(recommendations)} рекомендаций")
    
    # Анализируем результаты
    print_section("Анализ результатов")
    
    # Статистика по продуктам
    product_counts = pd.Series([rec['product'] for rec in recommendations]).value_counts()
    print("📈 Распределение рекомендаций по продуктам:")
    for product, count in product_counts.items():
        print(f"   {product}: {count} клиентов ({count/len(recommendations)*100:.1f}%)")
    
    # Статистика по уверенности
    confidences = [rec['confidence'] for rec in recommendations]
    print(f"\n🎯 Статистика уверенности:")
    print(f"   Средняя уверенность: {np.mean(confidences):.3f}")
    print(f"   Высокая уверенность (>80%): {sum(1 for c in confidences if c > 0.8)} клиентов")
    print(f"   Средняя уверенность (60-80%): {sum(1 for c in confidences if 0.6 <= c <= 0.8)} клиентов")
    print(f"   Низкая уверенность (<60%): {sum(1 for c in confidences if c < 0.6)} клиентов")
    
    # Статистика по кластерам
    cluster_counts = pd.Series([rec['cluster'] for rec in recommendations]).value_counts()
    print(f"\n🔍 Распределение по кластерам:")
    for cluster, count in cluster_counts.items():
        print(f"   Кластер {cluster}: {count} клиентов")
    
    # Топ рекомендации
    print_section("Топ-5 рекомендаций с высокой уверенностью")
    top_recommendations = sorted(recommendations, key=lambda x: x['confidence'], reverse=True)[:5]
    
    for i, rec in enumerate(top_recommendations, 1):
        print(f"\n{i}. {rec['name']} (ID: {rec['client_code']})")
        print(f"   Продукт: {rec['product']}")
        print(f"   Уверенность: {rec['confidence']:.1%}")
        print(f"   Выгода: {rec['benefit']:,.0f} ₸")
        print(f"   Кластер: {rec['cluster']}")
        print(f"   Сообщение: {rec['push_notification'][:100]}...")
    
    # Сравнение с классическим подходом
    print_section("Сравнение ML vs Классический подход")
    
    # Загружаем классические рекомендации для сравнения
    try:
        with open('all_recommendations.json', 'r', encoding='utf-8') as f:
            classic_recommendations = json.load(f)
        
        classic_products = [rec['product'] for rec in classic_recommendations]
        ml_products = [rec['product'] for rec in recommendations]
        
        print("📊 Сравнение распределения продуктов:")
        print("   Классический подход:")
        classic_counts = pd.Series(classic_products).value_counts()
        for product, count in classic_counts.items():
            print(f"     {product}: {count} ({count/len(classic_products)*100:.1f}%)")
        
        print("   ML подход:")
        ml_counts = pd.Series(ml_products).value_counts()
        for product, count in ml_counts.items():
            print(f"     {product}: {count} ({count/len(ml_products)*100:.1f}%)")
        
        # Анализ различий
        print("\n🔄 Анализ различий:")
        different_recommendations = 0
        for i, (classic, ml) in enumerate(zip(classic_products, ml_products)):
            if classic != ml:
                different_recommendations += 1
        
        print(f"   Различных рекомендаций: {different_recommendations}/{len(classic_products)} ({different_recommendations/len(classic_products)*100:.1f}%)")
        
    except FileNotFoundError:
        print("   Классические рекомендации не найдены для сравнения")
    
    # Сохраняем результаты
    print_section("Сохранение результатов")
    
    # Сохраняем ML рекомендации
    ml_df = pd.DataFrame(recommendations)
    ml_df.to_csv('ml_recommendations_demo.csv', index=False, encoding='utf-8-sig')
    print("✅ ML рекомендации сохранены в ml_recommendations_demo.csv")
    
    # Сохраняем детальный отчет
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_clients': len(recommendations),
        'accuracy': accuracy,
        'clusters_count': len(set(clusters)),
        'features_count': len(features_df.columns),
        'product_distribution': product_counts.to_dict(),
        'confidence_stats': {
            'mean': float(np.mean(confidences)),
            'high_confidence': int(sum(1 for c in confidences if c > 0.8)),
            'medium_confidence': int(sum(1 for c in confidences if 0.6 <= c <= 0.8)),
            'low_confidence': int(sum(1 for c in confidences if c < 0.6))
        },
        'cluster_distribution': cluster_counts.to_dict(),
        'top_recommendations': top_recommendations[:10]
    }
    
    with open('ml_analysis_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("✅ Детальный отчет сохранен в ml_analysis_report.json")
    
    # Финальная сводка
    print_header("Итоговая сводка")
    print("🎯 ML анализ успешно завершен!")
    print(f"📊 Проанализировано: {len(recommendations)} клиентов")
    print(f"🧠 Точность модели: {accuracy:.1%}")
    print(f"🔍 Обнаружено кластеров: {len(set(clusters))}")
    print(f"⚙️ Создано признаков: {len(features_df.columns)}")
    print(f"💡 Высокая уверенность: {sum(1 for c in confidences if c > 0.8)} рекомендаций")
    
    print("\n🚀 Готово для демонстрации на хакатоне!")
    print("   - Классические алгоритмы + ML")
    print("   - Персонализированные рекомендации")
    print("   - Кластеризация клиентов")
    print("   - Высокая точность предсказаний")

if __name__ == "__main__":
    main()
