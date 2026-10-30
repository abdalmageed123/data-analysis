import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    print("==================================================")
    print("      Week 3 Lab — Statistics I: Describing Data  ")
    print("==================================================\n")

    # تحميل البيانات
    try:
        orders = pd.read_csv("orders_data.csv")
        customers = pd.read_csv("customers.csv")
        daily = pd.read_csv("daily_sales.csv")
        print("✅ تم تحميل جميع الملفات بنجاح!\n")
    except FileNotFoundError as e:
        print(f"⚠️ خطأ: تعذر العثور على أحد ملفات البيانات ({e.filename}).")
        print("تأكد من وجود الملفات التالية في نفس المجلد: orders_data.csv, customers.csv, daily_sales.csv")
        return

    # ==========================================
    # Part 1 — Population, sample, and data
    # ==========================================
    print("--- Part 1: Population, Sample, and Data ---")
    print("1.1 الإجابات التحليلية لأسئلة المقاييس:")
    print("  1) Valid — متغير سلة المبيعات كمي نسبي (Ratio) ويمكن المقارنة بالنسبة المئوية.")
    print("  2) Invalid — المقياس المئوي (°C) مقياس فترات (Interval) وليس نسبياً، فالصفر لا يعني انعدام الحرارة.")
    print("  3) Invalid — رقم الطلب order_id متغير اسمي/معرف (Nominal ID) ولا معنى لحساب متوسطه.")
    print("  4) Valid — التقييم الترتيبي (Ordinal) يمكن حساب الوسيط (Median) له منطقياً.")
    print("  5) Invalid — طرق الدفع اسمية (Nominal)، الترميز الرقمي لا يجيز حساب المتوسط.\n")

    riyadh_median = orders[orders["city"] == "Riyadh"]["basket_value"].median()
    true_median = orders["basket_value"].median()
    pct_overestimate = (riyadh_median - true_median) / true_median * 100

    print(f"1.2 وسيط الرياض: ${riyadh_median:.2f} | الوسيط الحقيقي لكل المدن: ${true_median:.2f}")
    print(f"    نسبة المبالغة في التقدير: {pct_overestimate:.1f}%")
    print("    ✍️ نوع العينة: عينة ميسرة (Convenience Sample). التقدير مرتفع لأن الرياض ذات قوة شرائية أكبر.")
    print("    وكان ينبغي استخدام عينة عشوائية طبقية (Stratified Random Sample) تمثل جميع المدن.\n")

    # ==========================================
    # Part 2 — Frequency distributions and charts
    # ==========================================
    print("--- Part 2: Frequency Distributions and Charts ---")
    freq = orders["category"].value_counts()
    rel_freq = orders["category"].value_counts(normalize=True)
    cum_freq = rel_freq.cumsum()

    freq_table = pd.DataFrame({
        "frequency": freq,
        "relative": rel_freq.round(3),
        "cumulative": cum_freq.round(3)
    })
    print("جدول التكرار للفئات:")
    print(freq_table)
    
    top_mode = freq.index[0]
    top_3_share = rel_freq.iloc[:3].sum() * 100
    print(f"\n- المنوال (Mode): {top_mode}")
    print(f"- حصة أعلى 3 فئات: {top_3_share:.1f}%\n")

    # رسم الأشكال البيانية للقسم الثاني
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    freq.plot(kind="bar", color="skyblue", edgecolor="black")
    plt.title("Orders by Category")
    plt.xlabel("Category")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    plt.hist(orders["basket_value"], bins=25, color="salmon", edgecolor="black")
    plt.title("Basket Value Histogram")
    plt.xlabel("Basket Value ($)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    print("2.2 وصف الشكل: التوزيع ملتوٍ نحو اليمين (Right-skewed) مع وجود أقيم متطرفة مرتفعة.")
    print("2.3 سبب الذيل الأيمن الطويل: طلبات الإلكترونيات عالية القيمة يخلق هذا الذيل.")
    print("2.4 مقترح حذف الفئات الصغيرة: لا أؤيد الحذف لأنها تجلب نسبيًا عدد طلبات كبير وتجذب الزبائن للمتجر.\n")

    # ==========================================
    # Part 3 — Numerical summaries
    # ==========================================
    print("--- Part 3: Numerical Summaries ---")
    mean_val = orders["basket_value"].mean()
    median_val = orders["basket_value"].median()
    skew_val = orders["basket_value"].skew()
    print(f"Mean: {mean_val:.2f} | Median: {median_val:.2f} | Skewness: {skew_val:.2f}")
    print("✍️ نختار الوسيط (Median) للتعبير عن السلة النموذجية لأنه غير متأثر بالالتواء والقيم المتطرفة.\n")

    cat_summary = orders.groupby("category")["basket_value"].agg(["mean", "std"])
    cat_summary["cv"] = cat_summary["std"] / cat_summary["mean"]
    print("تشتت الفئات (الانحراف المعياري ومعامل الاختلاف CV):")
    print(cat_summary.sort_values("cv", ascending=False).round(2))
    print("\n✍️ نستخدم معامل الاختلاف (CV) للمقارنة بين الفئات ذات مستويات الأسعار المختلفة.\n")

    # حساب Outliers بـ IQR
    Q1 = orders["basket_value"].quantile(0.25)
    Q3 = orders["basket_value"].quantile(0.75)
    IQR = Q3 - Q1
    upper_fence = Q3 + 1.5 * IQR
    is_outlier = orders["basket_value"] > upper_fence

    print(f"الحد الأعلى للقيم المتطرفة (Upper Fence): {upper_fence:.2f}")
    print(f"عدد القيم المتطرفة: {is_outlier.sum()}")
    print("✍️ لا يجب حذف القيم المتطرفة لأنها تمثل طلبات حقيقية عالية الإيراد (مثل الإلكترونيات).\n")

    # ==========================================
    # Part 4 — Relationships between variables
    # ==========================================
    print("--- Part 4: Relationships Between Variables ---")
    ct_counts = pd.crosstab(customers["member"], customers["returned"], margins=True)
    ct_rows = pd.crosstab(customers["member"], customers["returned"], normalize="index").round(3)
    print("جدول التكرار المتقاطع (الأعضاء والعودة):")
    print(ct_rows)
    print("✍️ جدول نسب الأسطر يثبت أن نسبة عودة الأعضاء أعلى بكثير من غير الأعضاء.\n")

    print("مصفوفة الارتباط للبيانات اليومية:")
    print(daily.corr(numeric_only=True).round(3))
    print("\n")

    # ==========================================
    # Part 5 — Putting it together
    # ==========================================
 

    print("\n=== Part 5: Payment Methods Comparison ===")
    # استخدام اسم العمود الصحيح 'payment' الموجود في الملف
    payment_summary = orders.groupby("payment")["basket_value"].agg(["count", "mean", "median", "std"]).round(2)
    print(payment_summary)

    # رسم مقارنة قيم السلة حسب طريقة الدفع
    orders.boxplot(column="basket_value", by="payment")
    plt.ylabel("Basket Value ($)")
    plt.title("Basket Value by Payment Method")
    plt.suptitle("")
    plt.show()

    print("\n✍️ التوصية لفريق التسويق:")
    print("العملاء الذين يدفعون بالبطاقات والرقميات ينفقون متوسط سلة أعلى من الدفع النقدي.")
    print("نوصي بتقديم حوافز لتشجيع التحول للدفع الإلكتروني للرفع من قيمة السلة المبيعة.\n")

    print("==================================================")
    print("           تم تنفيذ جميع الأجزاء بنجاح!          ")
    print("==================================================")
if __name__ == "__main__":
    main()