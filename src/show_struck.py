from src.lstm_gan3 import *
from torchsummary import summary

if __name__ == "__main__":
    generator = Generator()
    summary(generator, (in_step, channels, image_size, image_size))
