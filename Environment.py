import pygame
import random
import os

pygame.init()
ROOT_DIR = os.path.dirname(__file__)


class Bird:
    """
    Bird class representing the flappy bird
    """

    def __init__(self, x, y):
        """
        Initialize the object
        :param x: starting x pos (int)
        :param y: starting y pos (int)
        :return: None
        """
        self.x = x
        self.y = y
        self.gravity = 1.5
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.bird_images = [pygame.image.load(os.path.realpath(os.path.join(ROOT_DIR, "assets/bird" + str(x) + ".png")))
                            for x in range(1, 4)]
        self.img = self.bird_images[0]

    def jump(self):
        """
        make the bird jump
        :return: None
        """
        self.vel = -5.25
        self.tick_count = 0
        self.height = self.y

    def move(self):
        """
        make the bird move
        :return: None
        """
        self.tick_count += 1

        # for displacement
        displacement = self.vel * self.tick_count + 0.5 * self.gravity * (
                    self.tick_count ** 2)  # calculate displacement

        if displacement >= 8:
            displacement = 8

        if displacement < 0:
            displacement -= 1

        self.y = self.y + displacement

    def draw(self, win):
        """
        draw the bird
        :param win: pygame window or surface
        :return: None
        """
        win.blit(self.img, (self.x, self.y))

    def get_mask(self):
        """
        gets the mask for the current image of the bird
        :return: None
        """
        return pygame.mask.from_surface(self.img)


class Pipe:
    """
    represents a pipe object
    """

    def __init__(self, x):
        """
        initialize pipe object
        :param x: int
        :return" None
        """
        self.x = x
        self.height = 0
        self.VEL = 5
        self.GAP = 100  # gap between top and bottom pipe

        # where the top and bottom of the pipe is
        self.top = 0
        self.bottom = 0
        self.pipe_img = pygame.image.load(os.path.realpath(os.path.join(ROOT_DIR, "assets/pipe.png")))

        self.PIPE_TOP = pygame.transform.flip(self.pipe_img, False, True)
        self.PIPE_BOTTOM = self.pipe_img

        self.passed = False

        self.set_height()

    def set_height(self):
        """
        set the height of the pipe, from the top of the screen
        :return: None
        """
        self.height = random.randrange(25, 225)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP

    def move(self):
        """
        move pipe based on vel
        :return: None
        """
        self.x -= self.VEL

    def draw(self, win):
        """
        draw both the top and bottom of the pipe
        :param win: pygame window/surface
        :return: None
        """
        # draw top
        win.blit(self.PIPE_TOP, (self.x, self.top))
        # draw bottom
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def collide(self, bird):
        """
        returns if a point is colliding with the pipe
        :param bird: Bird object
        :return: Bool
        """
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)
        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask, top_offset)

        if b_point or t_point:
            return True

        return False


class Base:
    """
    Represents the moving floor of the game
    """

    def __init__(self):
        """
        Initialize the object
        :return: None
        """
        self.y = 365
        self.x1 = 0
        self.IMG = pygame.image.load(os.path.realpath(os.path.join(ROOT_DIR, "assets/base.png")))
        self.WIDTH = self.IMG.get_width()
        self.x2 = self.WIDTH
        self.VEL = 5

    def move(self):
        """
        move floor so it looks like its scrolling
        :return: None
        """
        self.x1 -= self.VEL
        self.x2 -= self.VEL
        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH

        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, win):
        """
        Draw the floor. This is two images that move together.
        :param win: the pygame surface/window
        :return: None
        """
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))


def blitRotateCenter(win, image, topleft, angle):
    """
    Rotate a surface and blit it to the window
    :param win: the surface to blit to
    :param image: the image surface to rotate
    :param topleft: the top left position of the image
    :param angle: a float value for angle
    :return: None
    """
    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(center = image.get_rect(topleft = topleft).center)
    win.blit(rotated_image, new_rect.topleft)


class FlappyBirdEnv(object):
    pygame.display.init()
    win = pygame.display.set_mode((300, 400), flags = pygame.HIDDEN)
    pygame.display.set_caption("Flappy Bird")

    def __init__(self):
        self.WIN_WIDTH = 300
        self.WIN_HEIGHT = 400
        self.FLOOR = 365
        self.bird = Bird(115, 175)
        self.base = Base()
        self.pipes = [Pipe(275)]
        self.score = 0
        self.clock = pygame.time.Clock()
        self.lost = False
        self.action_space = [0, 1]
        self.observation_space = (self.WIN_WIDTH, self.WIN_HEIGHT, 3)

    def render(self):
        pygame.display.init()
        FlappyBirdEnv.win = pygame.display.set_mode((self.WIN_WIDTH, self.WIN_HEIGHT))
        pygame.display.set_caption("Flappy Bird")

    def Stop_render(self):
        pygame.display.init()
        FlappyBirdEnv.win = pygame.display.set_mode((self.WIN_WIDTH, self.WIN_HEIGHT), flags = pygame.HIDDEN)
        pygame.display.set_caption("Flappy Bird")

    def step(self, action):
        pygame.event.pump()
        self.clock.tick(30)
        reward = 0.1
        if action == 1:
            self.bird.jump()
        self.bird.move()
        self.base.move()
        rem = []
        add_pipe = False
        for pipe in self.pipes:
            pipe.move()
            # check for collision
            if pipe.collide(self.bird):
                reward = -1
                self.lost = True

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

            if not pipe.passed and pipe.x < self.bird.x:
                reward = 1
                pipe.passed = True
                add_pipe = True
            if (self.bird.y > pipe.height) and (self.bird.y < pipe.bottom):
                if self.bird.x + self.bird.img.get_width() == pipe.x:
                    reward = 0.5
                elif self.bird.x + self.bird.img.get_width() == pipe.x + (pipe.PIPE_TOP.get_width() / 2):
                    reward = 0.75
        if add_pipe:
            self.score += 1
            self.pipes.append(Pipe(self.WIN_WIDTH))
        for r in rem:
            self.pipes.remove(r)
        if self.bird.y + self.bird.bird_images[0].get_height() - 10 >= self.FLOOR or self.bird.y < -2:
            reward = -1
            self.lost = True
        image = self.draw_window()
        return image, reward, self.lost

    def reset(self, stop_render=True):
        self.bird = Bird(115, 175)
        self.base = Base()
        self.pipes = [Pipe(275)]
        self.score = 0
        if stop_render:
            self.Stop_render()
        self.clock = pygame.time.Clock()
        self.lost = False
        image = self.draw_window()
        return image

    def draw_window(self):
        """
        draws the windows for the main game loop
        :return: image
        """
        # win.blit(bg_img, (0, 0))
        self.win.fill((0, 0, 0))

        for pipe in self.pipes:
            pipe.draw(self.win)

        self.base.draw(self.win)
        self.bird.draw(self.win)
        pygame.display.update()
        array = pygame.surfarray.array3d(self.win)
        return array
