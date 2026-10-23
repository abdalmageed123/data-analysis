import numpy as np
import pandas as pd

print("=== Part 1: NumPy Warm-up ===")
price_list = [12.5, 45.0, 7.99, 88.2, 23.1]
unit_prices = np.array(price_list)
print("Shape:", unit_prices.shape, "| Dtype:", unit_prices.dtype)

# خصم 10% دفعة واحدة (Vectorized Operation)
discounted = unit_prices * 0.9
print("Discounted prices:", discounted)

# قناع بولياني للأسعار أكبر من 20
mask = unit_prices > 20
print("Prices > 20:", unit_prices[mask])

print("\n=== Part 2 & 3: Loading & Filtering Sales ===")
# تحميل بيانات المبيعات
sales = pd.read_csv('sales_data.csv')
print("Dataset Shape:", sales.shape)

# تحويل التاريخ واستخراج معلومات الأيام والأشهر
sales['date'] = pd.to_datetime(sales['date'])
sales['day_name'] = sales['date'].dt.day_name()

# تصفية مبيعات المغرب
morocco_sales = sales[sales['region'] == "Morocco"]
print("Morocco rows count:", len(morocco_sales))

print("\n=== Part 4: Grouping ===")
# إجمالي الإيرادات لكل منطقة
total_revenue_by_region = sales.groupby('region')['revenue'].sum()
print("Total revenue by region:\n", total_revenue_by_region)

print("\n=== Part 5: Basic Manipulation ===")
# إضافة عمود الإيرادات لكل وحدة وتصنيف حجم الطلب
sales['revenue_per_unit'] = sales['revenue'] / sales['units_sold']

def size_label(units):
    if units < 10:
        return "small"
    elif units <= 25:
        return "medium"
    else:
        return "large"

sales['order_size'] = sales['units_sold'].apply(size_label)
print("Order size counts:\n", sales['order_size'].value_counts())

print("\n=== Part 6: Finding Structure ===")
# التخصصات الإقليمية (أعلى منتج مبيعاً في كل منطقة)
reg_prod = sales.groupby(['region', 'product'], as_index=False)['revenue'].sum()
regional_specialties = reg_prod.sort_values(['region', 'revenue'], ascending=[True, False]).groupby('region').first()
print("Regional Specialties:\n", regional_specialties)

print("\n=== Part 7: Challenge Question ===")
# هل حقق عطر العود في الرياض مبيعات أعلى لكل طلب في عطلة نهاية الأسبوع؟
riyadh_oud = sales[(sales['region'] == 'Riyadh') & (sales['product'] == 'Oud Perfume')]
riyadh_oud['is_weekend'] = riyadh_oud['day_name'].isin(['Friday', 'Saturday'])
challenge_result = riyadh_oud.groupby('is_weekend').agg(
    avg_revenue_per_order=('revenue', 'mean'),
    order_count=('revenue', 'count')
)
print("Riyadh Oud Perfume Weekend vs Weekday:\n", challenge_result)