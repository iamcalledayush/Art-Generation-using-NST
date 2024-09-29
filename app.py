import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG19

# Define functions for NST (using your existing code)
def compute_content_cost(content_output, generated_output):
    a_C = content_output[-1]
    a_G = generated_output[-1]
    _, n_H, n_W, n_C = a_G.get_shape().as_list()
    a_C_unrolled = tf.reshape(a_C, shape=[_, -1, n_C])
    a_G_unrolled = tf.reshape(a_G, shape=[_, -1, n_C])
    J_content = tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled))) / (4 * n_H * n_W * n_C)
    return J_content

def gram_matrix(A):
    return tf.linalg.matmul(A, tf.transpose(A))

def compute_layer_style_cost(a_S, a_G):
    _, n_H, n_W, n_C = a_G.get_shape().as_list()
    a_S = tf.transpose(tf.reshape(a_S, shape=[-1, n_C]))
    a_G = tf.transpose(tf.reshape(a_G, shape=[-1, n_C]))
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)
    J_style_layer = (1 / (4 * n_C**2 * (n_H * n_W)**2)) * tf.reduce_sum(tf.square(tf.subtract(GS, GG)))
    return J_style_layer

STYLE_LAYERS = [
    ('block1_conv1', 0.2),
    ('block2_conv1', 0.2),
    ('block3_conv1', 0.2),
    ('block4_conv1', 0.2),
    ('block5_conv1', 0.2)
]

def compute_style_cost(style_image_output, generated_image_output, STYLE_LAYERS=STYLE_LAYERS):
    J_style = 0
    a_S = style_image_output[:-1]
    a_G = generated_image_output[:-1]
    for i, weight in zip(range(len(a_S)), STYLE_LAYERS):
        J_style_layer = compute_layer_style_cost(a_S[i], a_G[i])
        J_style += weight[1] * J_style_layer
    return J_style

def total_cost(J_content, J_style, alpha=10, beta=40):
    J = alpha * J_content + beta * J_style
    return J

def get_layer_outputs(vgg, layer_names):
    outputs = [vgg.get_layer(layer[0]).output for layer in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model

# Define Streamlit interface
st.title("Neural Style Transfer Web Interface")

# Upload content and style images
content_file = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"])
style_file = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"])

if content_file and style_file:
    content_image = Image.open(content_file)
    style_image = Image.open(style_file)

    st.image(content_image, caption='Content Image', use_column_width=True)
    st.image(style_image, caption='Style Image', use_column_width=True)

    # Define image size and convert to arrays
    img_size = 400
    content_image = content_image.resize((img_size, img_size))
    style_image = style_image.resize((img_size, img_size))
    
    content_image_array = np.array(content_image)
    style_image_array = np.array(style_image)

    # Convert to tensor
    content_image_tensor = tf.constant(np.reshape(content_image_array, ((1,) + content_image_array.shape)))
    style_image_tensor = tf.constant(np.reshape(style_image_array, ((1,) + style_image_array.shape)))

    # Display option to generate image
    if st.button("Generate Image"):
        # Load pre-trained VGG model
        vgg = VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False

        # Get model outputs for content and style
        content_layer = [('block5_conv4', 1)]
        all_layer_names = STYLE_LAYERS + content_layer
        vgg_model_outputs = get_layer_outputs(vgg, all_layer_names)
        
        content_target = vgg_model_outputs(content_image_tensor)
        style_targets = vgg_model_outputs(style_image_tensor)

        # Create a variable for the generated image
        generated_image = tf.Variable(tf.image.convert_image_dtype(content_image_tensor, tf.float32))
        
        # Train the model
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

        @tf.function()
        def train_step(generated_image):
            with tf.GradientTape() as tape:
                a_G = vgg_model_outputs(generated_image)
                J_style = compute_style_cost(style_targets, a_G)
                J_content = compute_content_cost(content_target, a_G)
                J = total_cost(J_content, J_style)
            grad = tape.gradient(J, generated_image)
            optimizer.apply_gradients([(grad, generated_image)])
            generated_image.assign(tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0))
            return J

        # Run the training process
        epochs = 500
        for i in range(epochs):
            train_step(generated_image)
            if i % 100 == 0:
                st.write(f"Epoch {i}: Style transfer in progress...")

        # Convert tensor to image and display
        generated_image_np = generated_image.numpy()[0]
        generated_image_pil = Image.fromarray((generated_image_np * 255).astype(np.uint8))
        st.image(generated_image_pil, caption='Generated Image', use_column_width=True)
