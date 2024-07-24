import gradio as gr
from huggingface_hub import InferenceClient

"""
For more information on `huggingface_hub` Inference API support, please check the docs: https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
"""
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")


def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    system_message = "Hello and welcome! I'm here to help you find your next great read. Tell me about the genres, authors, or types of stories you enjoy, and I'll recommend some books for you. Let's get started!"
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    response = ""

    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message.choices[0].delta.content

        response += token
        yield response

"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""

custom_css = """
body {background-color: #f9f9f9;}
h1 {color: #333; text-align: center; font-family: 'Arial', sans-serif;}
.gr-button {background-color: #4CAF50; color: white; border: none; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; transition-duration: 0.4s; cursor: pointer; border-radius: 16px;}
.gr-button:hover {background-color: white; color: black; border: 2px solid #4CAF50;}
"""

demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="Hello and welcome! I'm here to help you find your next great read. Tell me about the genres, authors, or types of stories you enjoy, and I'll recommend some books for you. Let's get started!", label="System Message", lines=3),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Maximum Tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (Nucleus Sampling)",
        ),
    ],

    examples=[
        ["I love mystery novels with a strong female lead."],
        ["Can you recommend some science fiction books?"],
        ["I'm looking for a good historical fiction novel."]
    ],
    title='📚 Personalized Book Recommendation Bot 📚',
    description='''<h2>Welcome to the Personalized Book Recommendation Bot!</h2>
                   <p>Tell me about your reading preferences, and I will suggest some books that you might enjoy.</p>
                   <p><strong>Examples:</strong></p>
                   <ul>
                       <li>I love mystery novels with a strong female lead.</li>
                       <li>Can you recommend some science fiction books?</li>
                       <li>I'm looking for a good historical fiction novel.</li>
                   </ul>''',
    css=custom_css
)


if __name__ == "__main__":
    demo.launch()
