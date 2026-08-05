# Starter redesigned app.py
import gradio as gr
CUSTOM_CSS=open("style.css").read()
with gr.Blocks(css=CUSTOM_CSS) as demo:
    gr.HTML("<div class='hero'><h1>AI Nutrition Plate Analyzer</h1><p>Clean card-based interface</p></div>")
    with gr.Row():
        with gr.Column():
            image=gr.Image(type='pil',label='Upload Food Image',height=320)
            age=gr.Number(label='Age')
            height=gr.Number(label='Height')
            weight=gr.Number(label='Weight')
            btn=gr.Button('Analyze Food')
        with gr.Column():
            gr.HTML("""<div class='result-card'><div class='header'><h2>Prediction</h2><span>92.3%</span></div><div class='stats'><div class='card'>Calories</div><div class='card'>Protein</div><div class='card'>Carbs</div><div class='card'>Fat</div><div class='card'>Fiber</div></div></div>""")
    # connect your existing analyze_food callback here
demo.launch()
