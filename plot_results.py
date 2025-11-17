import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 8)
plt.rcParams['font.size'] = 10

def load_data():
    """Load processed data and metrics"""
    print("📊 Loading data...")
    df = pd.read_csv("processed_data.csv")
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    metrics = pd.read_csv("metrics.csv").iloc[0]
    return df, metrics

def plot_last_6_months_comparison(df):
    """Plot last 6 months: Actual vs Predicted"""
    print("📈 Creating 6-month comparison plot...")
    
    # Get last 6 months of data
    last_date = df['Datetime'].max()
    six_months_ago = last_date - timedelta(days=180)
    df_6m = df[df['Datetime'] >= six_months_ago].copy()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    # Plot 1: Time series comparison
    ax1.plot(df_6m['Datetime'], df_6m['Actual_Power'], 
             label='Actual Power', color='#2E86AB', linewidth=1.5, alpha=0.8)
    ax1.plot(df_6m['Datetime'], df_6m['LSTM_Prediction'], 
             label='LSTM+GAN Prediction', color='#A23B72', linewidth=1.5, alpha=0.8)
    ax1.fill_between(df_6m['Datetime'], df_6m['Actual_Power'], df_6m['LSTM_Prediction'], 
                      alpha=0.2, color='gray', label='Prediction Error')
    
    ax1.set_title('Last 6 Months: Actual vs Predicted Power Consumption', 
                  fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Power (KW)', fontsize=12)
    ax1.legend(loc='upper right', fontsize=11)
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 2: Error distribution over time
    df_6m['Error'] = df_6m['Actual_Power'] - df_6m['LSTM_Prediction']
    ax2.plot(df_6m['Datetime'], df_6m['Error'], color='#F18F01', linewidth=1, alpha=0.7)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    ax2.fill_between(df_6m['Datetime'], 0, df_6m['Error'], 
                      where=(df_6m['Error'] >= 0), alpha=0.3, color='green', label='Under-prediction')
    ax2.fill_between(df_6m['Datetime'], 0, df_6m['Error'], 
                      where=(df_6m['Error'] < 0), alpha=0.3, color='red', label='Over-prediction')
    
    ax2.set_title('Prediction Error Over Time', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Error (KW)', fontsize=12)
    ax2.legend(loc='upper right', fontsize=11)
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig("6_months_comparison.png", dpi=300, bbox_inches='tight')
    print("✅ Saved: 6_months_comparison.png")
    plt.close()

def plot_weekly_patterns(df):
    """Analyze weekly patterns"""
    print("📈 Creating weekly pattern analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Average power by day of week
    df['DayOfWeek'] = df['Datetime'].dt.day_name()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_avg = df.groupby('DayOfWeek').agg({
        'Actual_Power': 'mean',
        'LSTM_Prediction': 'mean'
    }).reindex(day_order)
    
    x = np.arange(len(day_order))
    width = 0.35
    axes[0, 0].bar(x - width/2, daily_avg['Actual_Power'], width, 
                   label='Actual', color='#2E86AB', alpha=0.8)
    axes[0, 0].bar(x + width/2, daily_avg['LSTM_Prediction'], width, 
                   label='Predicted', color='#A23B72', alpha=0.8)
    axes[0, 0].set_xlabel('Day of Week', fontsize=11)
    axes[0, 0].set_ylabel('Average Power (KW)', fontsize=11)
    axes[0, 0].set_title('Average Power Consumption by Day of Week', fontsize=13, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(day_order, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Hourly patterns
    df['Hour'] = df['Datetime'].dt.hour
    hourly_avg = df.groupby('Hour').agg({
        'Actual_Power': 'mean',
        'LSTM_Prediction': 'mean'
    })
    
    axes[0, 1].plot(hourly_avg.index, hourly_avg['Actual_Power'], 
                    marker='o', label='Actual', color='#2E86AB', linewidth=2)
    axes[0, 1].plot(hourly_avg.index, hourly_avg['LSTM_Prediction'], 
                    marker='s', label='Predicted', color='#A23B72', linewidth=2)
    axes[0, 1].set_xlabel('Hour of Day', fontsize=11)
    axes[0, 1].set_ylabel('Average Power (KW)', fontsize=11)
    axes[0, 1].set_title('Average Power Consumption by Hour', fontsize=13, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xticks(range(0, 24, 3))
    
    # Plot 3: Monthly trends
    df['Month'] = df['Datetime'].dt.to_period('M')
    monthly_avg = df.groupby('Month').agg({
        'Actual_Power': 'mean',
        'LSTM_Prediction': 'mean'
    })
    
    axes[1, 0].plot(range(len(monthly_avg)), monthly_avg['Actual_Power'], 
                    marker='o', label='Actual', color='#2E86AB', linewidth=2, markersize=8)
    axes[1, 0].plot(range(len(monthly_avg)), monthly_avg['LSTM_Prediction'], 
                    marker='s', label='Predicted', color='#A23B72', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Month', fontsize=11)
    axes[1, 0].set_ylabel('Average Power (KW)', fontsize=11)
    axes[1, 0].set_title('Monthly Average Power Consumption', fontsize=13, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xticks(range(len(monthly_avg)))
    axes[1, 0].set_xticklabels([str(m) for m in monthly_avg.index], rotation=45, ha='right')
    
    # Plot 4: Weekend vs Weekday
    df['IsWeekend'] = df['Datetime'].dt.dayofweek >= 5
    weekend_stats = df.groupby('IsWeekend').agg({
        'Actual_Power': ['mean', 'std'],
        'LSTM_Prediction': ['mean', 'std']
    })
    
    categories = ['Weekday', 'Weekend']
    actual_means = [weekend_stats.loc[False, ('Actual_Power', 'mean')], 
                   weekend_stats.loc[True, ('Actual_Power', 'mean')]]
    pred_means = [weekend_stats.loc[False, ('LSTM_Prediction', 'mean')], 
                 weekend_stats.loc[True, ('LSTM_Prediction', 'mean')]]
    actual_stds = [weekend_stats.loc[False, ('Actual_Power', 'std')], 
                  weekend_stats.loc[True, ('Actual_Power', 'std')]]
    pred_stds = [weekend_stats.loc[False, ('LSTM_Prediction', 'std')], 
                weekend_stats.loc[True, ('LSTM_Prediction', 'std')]]
    
    x = np.arange(len(categories))
    axes[1, 1].bar(x - width/2, actual_means, width, yerr=actual_stds,
                   label='Actual', color='#2E86AB', alpha=0.8, capsize=5)
    axes[1, 1].bar(x + width/2, pred_means, width, yerr=pred_stds,
                   label='Predicted', color='#A23B72', alpha=0.8, capsize=5)
    axes[1, 1].set_xlabel('Day Type', fontsize=11)
    axes[1, 1].set_ylabel('Average Power (KW)', fontsize=11)
    axes[1, 1].set_title('Weekday vs Weekend Power Consumption', fontsize=13, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(categories)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("weekly_patterns.png", dpi=300, bbox_inches='tight')
    print("✅ Saved: weekly_patterns.png")
    plt.close()

def plot_error_analysis(df):
    """Detailed error analysis"""
    print("📈 Creating error analysis plots...")
    
    df['Error'] = df['Actual_Power'] - df['LSTM_Prediction']
    df['Error_Percent'] = (df['Error'] / df['Actual_Power']) * 100
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Error distribution
    axes[0, 0].hist(df['Error'], bins=50, color='#F18F01', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(df['Error'].mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {df["Error"].mean():.2f}')
    axes[0, 0].axvline(0, color='green', linestyle='-', linewidth=2, alpha=0.5)
    axes[0, 0].set_xlabel('Prediction Error (KW)', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('Distribution of Prediction Errors', fontsize=13, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Actual vs Predicted scatter
    axes[0, 1].scatter(df['Actual_Power'], df['LSTM_Prediction'], 
                       alpha=0.3, s=10, color='#2E86AB')
    
    # Perfect prediction line
    min_val = min(df['Actual_Power'].min(), df['LSTM_Prediction'].min())
    max_val = max(df['Actual_Power'].max(), df['LSTM_Prediction'].max())
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 
                    'r--', linewidth=2, label='Perfect Prediction')
    
    axes[0, 1].set_xlabel('Actual Power (KW)', fontsize=11)
    axes[0, 1].set_ylabel('Predicted Power (KW)', fontsize=11)
    axes[0, 1].set_title('Actual vs Predicted Power (Scatter Plot)', fontsize=13, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Error percentage distribution
    axes[1, 0].hist(df['Error_Percent'], bins=50, color='#A23B72', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(df['Error_Percent'].mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {df["Error_Percent"].mean():.2f}%')
    axes[1, 0].set_xlabel('Prediction Error (%)', fontsize=11)
    axes[1, 0].set_ylabel('Frequency', fontsize=11)
    axes[1, 0].set_title('Distribution of Percentage Errors', fontsize=13, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Error by power level
    df['Power_Bin'] = pd.cut(df['Actual_Power'], bins=10)
    error_by_power = df.groupby('Power_Bin')['Error'].agg(['mean', 'std'])
    
    x_pos = range(len(error_by_power))
    axes[1, 1].bar(x_pos, error_by_power['mean'], yerr=error_by_power['std'],
                   color='#F18F01', alpha=0.7, capsize=5)
    axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=1.5)
    axes[1, 1].set_xlabel('Power Level Bins', fontsize=11)
    axes[1, 1].set_ylabel('Average Error (KW)', fontsize=11)
    axes[1, 1].set_title('Prediction Error by Power Level', fontsize=13, fontweight='bold')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels([f'{i+1}' for i in x_pos], rotation=0)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("error_analysis.png", dpi=300, bbox_inches='tight')
    print("✅ Saved: error_analysis.png")
    plt.close()

def plot_metrics_summary(metrics):
    """Display metrics in a visual format"""
    print("📈 Creating metrics summary...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: LSTM vs Refined metrics comparison
    metric_names = ['MSE', 'RMSE', 'MAE']
    lstm_values = [metrics['LSTM_MSE'], metrics['LSTM_RMSE'], metrics['LSTM_MAE']]
    refined_values = [metrics['Refined_MSE'], metrics['Refined_RMSE'], metrics['Refined_MAE']]
    
    x = np.arange(len(metric_names))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, lstm_values, width, label='LSTM Only', 
                    color='#2E86AB', alpha=0.8)
    bars2 = ax1.bar(x + width/2, refined_values, width, label='LSTM + GAN', 
                    color='#A23B72', alpha=0.8)
    
    ax1.set_xlabel('Metrics', fontsize=12)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.set_title('Model Performance Comparison (Scaled)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_names)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: R² Score comparison
    r2_data = {
        'LSTM Only': metrics['LSTM_R2'],
        'LSTM + GAN\n(Scaled)': metrics['Refined_R2'],
        'LSTM + GAN\n(Actual)': metrics['Actual_R2']
    }
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    bars = ax2.bar(range(len(r2_data)), list(r2_data.values()), 
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax2.set_xlabel('Model Configuration', fontsize=12)
    ax2.set_ylabel('R² Score', fontsize=12)
    ax2.set_title('R² Score Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(r2_data)))
    ax2.set_xticklabels(list(r2_data.keys()), fontsize=10)
    ax2.set_ylim([0, 1])
    ax2.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, linewidth=1.5, label='Good Threshold (0.8)')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(fontsize=10)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, r2_data.values())):
        ax2.text(bar.get_x() + bar.get_width()/2., value + 0.02,
                f'{value:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("metrics_summary.png", dpi=300, bbox_inches='tight')
    print("✅ Saved: metrics_summary.png")
    plt.close()

def print_metrics_table(metrics):
    """Print formatted metrics table"""
    print("\n" + "="*70)
    print("📊 MODEL PERFORMANCE METRICS")
    print("="*70)
    
    print("\n🔹 LSTM-Only Performance (Scaled):")
    print(f"   MSE:  {metrics['LSTM_MSE']:.6f}")
    print(f"   RMSE: {metrics['LSTM_RMSE']:.6f}")
    print(f"   MAE:  {metrics['LSTM_MAE']:.6f}")
    print(f"   R²:   {metrics['LSTM_R2']:.6f}")
    
    print("\n🔹 LSTM + GAN Performance (Scaled):")
    print(f"   MSE:  {metrics['Refined_MSE']:.6f}")
    print(f"   RMSE: {metrics['Refined_RMSE']:.6f}")
    print(f"   MAE:  {metrics['Refined_MAE']:.6f}")
    print(f"   R²:   {metrics['Refined_R2']:.6f}")
    
    print("\n🔹 LSTM + GAN Performance (Actual Scale):")
    print(f"   MSE:  {metrics['Actual_MSE']:.2f}")
    print(f"   RMSE: {metrics['Actual_RMSE']:.2f}")
    print(f"   MAE:  {metrics['Actual_MAE']:.2f}")
    print(f"   R²:   {metrics['Actual_R2']:.6f}")
    
    # Improvement calculation
    improvement = ((metrics['LSTM_R2'] - metrics['Refined_R2']) / metrics['LSTM_R2']) * 100
    if improvement < 0:
        print(f"\n✨ GAN improved R² by {abs(improvement):.2f}%")
    else:
        print(f"\n⚠️  GAN decreased R² by {improvement:.2f}%")
    
    print("="*70 + "\n")

def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("🎨 POWER CONSUMPTION PREDICTION - VISUALIZATION SUITE")
    print("="*70 + "\n")
    
    # Load data
    df, metrics = load_data()
    
    # Print metrics
    print_metrics_table(metrics)
    
    # Generate all plots
    plot_last_6_months_comparison(df)
    plot_weekly_patterns(df)
    plot_error_analysis(df)
    plot_metrics_summary(metrics)
    
    print("\n" + "="*70)
    print("✅ ALL VISUALIZATIONS COMPLETED!")
    print("="*70)
    print("\n📁 Generated Files:")
    print("   1. 6_months_comparison.png - Last 6 months actual vs predicted")
    print("   2. weekly_patterns.png - Daily, hourly, and monthly patterns")
    print("   3. error_analysis.png - Detailed error distribution and analysis")
    print("   4. metrics_summary.png - Performance metrics comparison")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()