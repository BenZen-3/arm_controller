# import time

# true_start = time.time()

# from sentence_transformers import SentenceTransformer

# print(f"huh {time.time() - true_start}")

# start = time.time()
# print('started')
# # 1. Load a pretrained Sentence Transformer model
# model = SentenceTransformer("all-MiniLM-L6-v2")

# print(f"model loaded at time {time.time() - start}")

# # The sentences to encode
# sentences = [
#     "The weather is lovely today.",
#     "It's so sunny outside!",
#     "He drove to the stadium.",
# ]

# # 2. Calculate embeddings by calling model.encode()
# embeddings = model.encode(sentences)
# print(embeddings.shape)

# print('end')

# # [3, 384]

# # # 3. Calculate the embedding similarities
# # similarities = model.similarity(embeddings, embeddings)
# # print(similarities)



import numpy as np

radius = 16
num_points = 12
angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

# Calculate the points on the circle
points = [(round(radius * np.cos(angle), 4), round(radius * np.sin(angle), 4)) for angle in angles]

print(points)

"""
I am prompting an LLM with the following prompt. What would help improve this prompt to make a higher quality dataset?

PROMPT:

Hello chat. I am training a small language conditioned diffusion model and I need to generate conditioned training data. The model predicts the movement of a 2DoF arm given text input. I would like you to help generate example text inputs and arm end effector positions in order to train this model.

Here are some rules that the generated data needs to follow. Please be sure to follow these rules. 
1) The text prompt should be relevant to the arm's movement.
2) The text prompt must be shorter than 256 word pieces in order to be processed by a language encoder.
3) The text prompts should be varied and creative enough to provide a diverse dataset.
4) The arm's speed is controllable in the range [1, 5] inclusive. 
5) The coordinates are end effector positions and should be floats. X coordinate in range [-32, +32] and Y in [-32, +32], with unspecified units. 
6) Although the acceptable coordinates span X [-32, 32] and Y [-32, 32], the arm has a more limited workspace. It is a 2DoF arm with maximum workspace radius of 30 units. For example, this would limit the arm to being unable to reach the point (25,25) due to the arm's limited reach. Please keep the coordinates inside the arm's workspace.
6) End effector coordinate points will be linearly interpolated between. Density of points should be determined by spatial considerations, not temporal. A separate controller will drive the arm between the points, so point density should be determined by the effects of timing.
7) The starting coordinate should be varied between prompts. The first coordinate in the list can be any point in the acceptable range and the desired motion should be fulfilled by the following coordinates. 

Here are three example responses. Please follow this format. Only respond with examples; do not respond with extra text. Thank you chat.

Example 1:
Text prompt: "Arm moves to the right fast"
Speed: 4
Coordinate list: [(0,-10), (5,-10), (10,-10), (15,-10)]

Example 2:
Text prompt: "Move the arm in a circular motion with radius 16 units slowly"
Speed: 2
Coordinate list: [(16.0, 0.0), (13.8564, 8.0), (8.0, 13.8564), (0.0, 16.0), (-8.0, 13.8564), (-13.8564, 8.0), (-16.0, 0.0), (-13.8564, -8.0), (-8.0, -13.8564), (-0.0, -16.0), (8.0, -13.8564), (13.8564, -8.0)]

Example 3:
Text prompt: "Square with side lengths of 10 centered around the arm"
Speed: 3
Coordinate list: [(10,10), (10,-10), (-10,-10), (-10,10)]
"""