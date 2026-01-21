import gradio as gr


def greet(name):
    return f'Hello {name}!'

def main():
    with gr.Blocks() as app:
        gr.Markdown("# Welcome to Hugging Face Spaces")
        name_input = gr.Textbox(label="Enter your name")
        greet_btn = gr.Button(label="Greet")
        greet_output = gr.Textbox(label="Greeting")

        greet_btn.click(greet, inputs=name_input, outputs=greet_output)

    app.launch()

if __name__ == '__main__':
    main()