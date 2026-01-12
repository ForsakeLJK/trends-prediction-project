import json
import os

# --- Configuration ---
REPORT_PATH = 'hindcast/trend_report.json'
IMAGE_PATH = 'hindcast/top_5_hit_rate.png'
OUTPUT_FILE = 'dashboard.html'

def load_data():
    if not os.path.exists(REPORT_PATH):
        return None
    with open(REPORT_PATH, 'r') as f:
        return json.load(f)

def generate_table_rows(data_list):
    if not data_list:
        return '<tr><td colspan="2" class="py-4 text-center text-gray-500">No data available</td></tr>'
    rows = ""
    for item in data_list:
        trend = item.get('trend', 'N/A')
        count = item.get('count', 0)
        # Formating count to 1 decimal if it's a float, else int
        display_count = f"{count:.1f}" if isinstance(count, float) else count
        rows += f"""
        <tr class="border-b border-gray-700/50 hover:bg-gray-800/50 transition-colors">
            <td class="py-3 px-4 text-blue-400 font-medium">{trend}</td>
            <td class="py-3 px-4 text-gray-300 text-right font-mono">{display_count}</td>
        </tr>
        """
    return rows

def main():
    data = load_data()
    if not data:
        print(f"Error: Could not find {REPORT_PATH}")
        return

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>BlueSky Forecast Dashboard</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap" rel="stylesheet">
        <style>
            body {{ 
                font-family: 'Plus Jakarta Sans', sans-serif; 
                background-color: #0b0f1a; 
                color: #f8fafc; 
            }}
            .glass {{ 
                background: rgba(23, 32, 53, 0.8); 
                backdrop-filter: blur(12px); 
                border: 1px solid rgba(255, 255, 255, 0.05); 
            }}
            .card-glow {{
                box-shadow: 0 0 20px rgba(59, 130, 246, 0.1);
            }}
        </style>
    </head>
    <body class="p-4 md:p-12">
        <div class="max-w-7xl mx-auto">
            
            <header class="mb-10 flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                    <div class="flex items-center gap-3 mb-2">
                        <div class="w-8 h-8 bg-blue-500 rounded-lg flex items-center justify-center shadow-lg shadow-blue-500/20">
                            <svg class="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"></path></svg>
                        </div>
                        <h1 class="text-3xl font-bold tracking-tight text-white">BlueSky Trend Analysis</h1>
                    </div>
                    <p class="text-gray-400">Hindcast Evaluation & Predictive Modeling Report</p>
                </div>
                <div class="flex gap-3">
                    <div class="glass px-4 py-2 rounded-xl text-sm border border-gray-700">
                        <span class="text-gray-500 mr-2">Status:</span>
                        <span class="text-emerald-400 font-semibold">● Ready</span>
                    </div>
                </div>
            </header>

            <section class="mb-8">
                <div class="glass rounded-3xl p-6 md:p-8 card-glow">
                    <div class="flex items-center justify-between mb-6">
                        <h2 class="text-xl font-semibold flex items-center gap-2">
                            <span class="text-blue-500">📈</span> Top 5 Hit Rate Performance
                        </h2>
                        <span class="text-xs font-mono text-gray-500 uppercase tracking-widest">Model Accuracy Hindcast</span>
                    </div>
                    <div class="bg-[#05070a] rounded-2xl p-4 border border-gray-800 shadow-inner">
                        <img src="{IMAGE_PATH}" alt="Hit Rate Chart" class="w-full h-auto max-h-[500px] object-contain mx-auto rounded-lg">
                    </div>
                </div>
            </section>

            <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                
                <div class="glass rounded-2xl p-6 border-t-4 border-gray-500">
                    <h3 class="text-sm font-bold text-gray-400 uppercase tracking-widest mb-6 flex justify-between">
                        Actual Trends <span class="text-gray-600">Now</span>
                    </h3>
                    <table class="w-full text-left">
                        <thead class="text-xs text-gray-500 uppercase">
                            <tr class="border-b border-gray-800">
                                <th class="pb-3 px-4">Tag</th>
                                <th class="pb-3 px-4 text-right">Count</th>
                            </tr>
                        </thead>
                        <tbody>
                            {generate_table_rows(data.get('current_actual_top5', []))}
                        </tbody>
                    </table>
                </div>

                <div class="glass rounded-2xl p-6 border-t-4 border-blue-500">
                    <h3 class="text-sm font-bold text-blue-400 uppercase tracking-widest mb-6 flex justify-between">
                        Predicted <span class="text-blue-900/50 italic font-medium">Validation</span>
                    </h3>
                    <table class="w-full text-left">
                        <thead class="text-xs text-gray-500 uppercase">
                            <tr class="border-b border-gray-800">
                                <th class="pb-3 px-4">Tag</th>
                                <th class="pb-3 px-4 text-right">Count</th>
                            </tr>
                        </thead>
                        <tbody>
                            {generate_table_rows(data.get('current_predicted_top5', []))}
                        </tbody>
                    </table>
                </div>

                <div class="glass rounded-2xl p-6 border-t-4 border-emerald-500 shadow-lg shadow-emerald-500/5">
                    <h3 class="text-sm font-bold text-emerald-400 uppercase tracking-widest mb-6 flex justify-between">
                        Future Forecast <span class="text-emerald-900/50 italic font-medium">Next 5 Minutes</span>
                    </h3>
                    <table class="w-full text-left">
                        <thead class="text-xs text-gray-500 uppercase">
                            <tr class="border-b border-gray-800">
                                <th class="pb-3 px-4">Tag</th>
                                <th class="pb-3 px-4 text-right">Count</th>
                            </tr>
                        </thead>
                        <tbody>
                            {generate_table_rows(data.get('next_predicted_top5', []))}
                        </tbody>
                    </table>
                </div>

            </div>

            <footer class="mt-12 py-6 text-center border-t border-gray-900">
                <p class="text-gray-600 text-sm italic">
                    Automated report generated from <strong>{REPORT_PATH}</strong>. All predictions are based on the BlueSky API firehose.
                </p>
            </footer>
        </div>
    </body>
    </html>
    """

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Dashboard successfully created: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()