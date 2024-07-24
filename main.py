import pygame
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import joblib

# initial game state
game_state = 'clear'
RADIUS = 10 # radius of drawing pen

# load model
# random_forest_classifier = joblib.load('../models/scaled_forest_classifier.joblib')
cnn = joblib.load('models/mnist_cnn.joblib')

# load image icon
icon = pygame.image.load('assets/pixel_pencil.jpg')

def modify_screen_data(screen_2d_array):

    # change values in 2d array
    for r in range(len(screen_2d_array)):
        for c in range(len(screen_2d_array[0])):
            
            if screen_2d_array[r][c] == 16777215:
                screen_2d_array[r][c] = 0
            else:
                screen_2d_array[r][c] = 1

    # pool data to make a 28 by 28 array for model to analyze
    def pool_array(arr, block_size):
        m, n = arr.shape
        pooled_arr = arr.reshape(m // block_size, block_size, -1, block_size).mean(axis=(1,3))
        return pooled_arr
    
    shrunk_array = pool_array(screen_2d_array, 10)
    shrunk_array = np.rot90(shrunk_array, k=3)
    shrunk_array = np.flip(shrunk_array, axis=1)
    return shrunk_array

# Function which displays the probablities of a prob array
def list_probs(probs):
    counter = 0
    for prob in probs:
        print(f'The probability for {counter} is {prob * 100}')
        counter += 1

    
def start():

    # pygame setup
    pygame.init()
    screen = pygame.display.set_mode((280, 280))
    clock = pygame.time.Clock()
    running = True
    global game_state
    draw_state = 'pen_up' # initialize pen as up
    pygame.display.set_icon(icon)

    def handle_draw():
        x, y = pygame.mouse.get_pos()
        pygame.draw.circle(screen, 'black', (x, y), RADIUS)

    while running:

        if game_state == 'clear':
            game_state = 'drawing'
            screen.fill('white')
            pygame.display.set_caption("Draw!")

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            # pen logic
            if event.type == pygame.MOUSEBUTTONDOWN:
                draw_state = 'pen_down'
            if event.type == pygame.MOUSEBUTTONUP:
                draw_state = 'pen_up'

            # drawing logic
            if event.type == pygame.MOUSEMOTION and draw_state == 'pen_down':
                handle_draw()

            if event.type == pygame.KEYDOWN:
                
                # clearing logic
                if event.key == pygame.K_c:
                    game_state = 'clear'

                # save image logic
                if event.key == pygame.K_RETURN:
                    screen_image = pygame.surfarray.array2d(screen)
                    shrunk_array = modify_screen_data(screen_image)

                    # SHOWING DRAWING
                    # plt.imshow(shrunk_array, cmap='binary')
                    # plt.show()

                    cnn_prep = shrunk_array.reshape(-1, 28, 28, 1)
                    predicted_val_probs = cnn.predict(cnn_prep)
                    list_probs(predicted_val_probs[0])
                    predicted_val = np.argmax(predicted_val_probs)
                    pygame.display.set_caption(f"{predicted_val}")
            

        pygame.display.update()
    
    pygame.quit()

if __name__ == '__main__':
    start()