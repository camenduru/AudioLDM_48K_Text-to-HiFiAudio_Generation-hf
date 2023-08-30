from sys import maxsize
from huggingface_hub import hf_hub_download
import torch
import os

import gradio as gr
from audioldm2 import text_to_audio, build_model

default_checkpoint="audioldm_48k"
audioldm = None
current_model_name = None

def text2audio(
    text,
    duration,
    guidance_scale,
    random_seed,
    n_candidates,
    model_name=default_checkpoint,
):
    global audioldm, current_model_name
    torch.set_float32_matmul_precision("high")

    if audioldm is None or model_name != current_model_name:
        audioldm = build_model(model_name=model_name)
        current_model_name = model_name
    if("48k" in model_name):
        latent_t_per_second=12.8
        sample_rate=48000
    else:
        latent_t_per_second=25.6
        sample_rate=16000

    waveform = text_to_audio(
        latent_diffusion=audioldm,
        text=text,
        seed=random_seed,
        duration=duration,
        guidance_scale=guidance_scale,
        n_candidate_gen_per_text=int(n_candidates),
        latent_t_per_second=latent_t_per_second,
    )
    waveform = [
        gr.make_waveform((sample_rate, wave[0]), bg_image="bg.png") for wave in waveform
    ]
    if len(waveform) == 1:
        waveform = waveform[0]
    return waveform

text2audio("Birds singing sweetly in a blooming garden.", 10, 3.5, 45, 3, default_checkpoint)
iface = gr.Blocks()

with iface:
    gr.HTML(
        """
            <div style="text-align: center; max-width: 700px; margin: 0 auto;">
              <div
                style="
                  display: inline-flex;
                  align-items: center;
                  gap: 0.8rem;
                  font-size: 1.75rem;
                "
              >
                <h1 style="font-weight: 900; margin-bottom: 7px; line-height: normal;">
                  48kHz AudioLDM: Generating High-Fidelity Audio and Music with Text
                </h1>
              </div>
              <p style="margin-bottom: 10px; font-size: 94%">
                <a href="https://arxiv.org/abs/2308.05734">[Paper]</a>  <a href="https://audioldm.github.io/audioldm2">[Project page]</a> <a href="https://discord.com/invite/b64SEmdf">[Join Discord]</a>
              </p>
            </div>
        """
    )
    with gr.Group():
        with gr.Box():
            textbox = gr.Textbox(
                value="A forest of wind chimes singing a soothing melody in the breeze.",
                max_lines=1,
                label="Input your text here. If the output is not good enough, switching to a different seed will help.",
                elem_id="prompt-in",
            )
            with gr.Accordion("Click to modify detailed configurations", open=False):
                seed = gr.Number(
                    value=45,
                    label="Change this value (any integer number) will lead to a different generation result.",
                )
                duration = gr.Slider(
                    5, 15, value=10, step=2.5, label="Duration (seconds)"
                )
                guidance_scale = gr.Slider(
                    0,
                    6,
                    value=3.5,
                    step=0.5,
                    label="Guidance scale (Large => better quality and relavancy to text; Small => better diversity)",
                )
                n_candidates = gr.Slider(
                    1,
                    3,
                    value=3,
                    step=1,
                    label="Automatic quality control. This number control the number of candidates (e.g., generate three audios and choose the best to show you). A Larger value usually lead to better quality with heavier computation",
                )
                model_name = gr.Dropdown(
                      ["audioldm_48k", "audioldm_crossattn_flant5", "audioldm2-full"], value="audioldm_48k",
                  )
            outputs = gr.Video(label="Output", elem_id="output-video")
            btn = gr.Button("Submit").style(full_width=True)

        btn.click(
            text2audio,
            inputs=[textbox, duration, guidance_scale, seed, n_candidates],
            outputs=[outputs],
            api_name="text2audio",
        )
        gr.HTML(
            """
        <div class="footer" style="text-align: center; max-width: 700px; margin: 0 auto;">
                    <p>Follow the latest update of AudioLDM 2 on our<a href="https://github.com/haoheliu/AudioLDM2" style="text-decoration: underline;" target="_blank"> Github repo</a>
                    </p>
                    <br>
                    <p>Model by <a href="https://twitter.com/LiuHaohe" style="text-decoration: underline;" target="_blank">Haohe Liu</a></p>
                    <br>
        </div>
        """
        )
        gr.Examples(
            [
                [
                    "Birds singing sweetly in a blooming garden.",
                    10,
                    3.5,
                    45,
                    3,
                    default_checkpoint,
                ],
                [
                    "A modern synthesizer creating futuristic soundscapes.",
                    10,
                    3.5,
                    45,
                    3,
                    default_checkpoint,
                ],
                [
                    "The vibrant beat of Brazilian samba drums.",
                    10,
                    3.5,
                    45,
                    3,
                    default_checkpoint,
                ],
            ],
            fn=text2audio,
            inputs=[textbox, duration, guidance_scale, seed, n_candidates, model_name],
            outputs=[outputs],
            cache_examples=False,
        )
        gr.HTML(
            """
                <div class="acknowledgements">
                <p>Essential Tricks for Enhancing the Quality of Your Generated Audio</p>
                <p>1. Try to use more adjectives to describe your sound. For example: "A man is speaking clearly and slowly in a large room" is better than "A man is speaking". This can make sure AudioLDM 2 understands what you want.</p>
                <p>2. Try to use different random seeds, which can affect the generation quality significantly sometimes.</p>
                <p>3. It's better to use general terms like 'man' or 'woman' instead of specific names for individuals or abstract objects that humans may not be familiar with, such as 'mummy'.</p>
                </div>
                """
        )

        with gr.Accordion("Additional information", open=False):
            gr.HTML(
                """
                <div class="acknowledgments">
                    <p> We build the model with data from <a href="http://research.google.com/audioset/">AudioSet</a>, <a href="https://freesound.org/">Freesound</a> and <a href="https://sound-effects.bbcrewind.co.uk/">BBC Sound Effect library</a>. We share this demo based on the <a href="https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/375954/Research.pdf">UK copyright exception</a> of data for academic research. </p>
                            </div>
                        """
            )

iface.queue(max_size=20)
iface.launch(debug=True, share=True)
