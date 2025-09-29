import pygame
import math
import sys
import numpy as np

import random

import ia

show = True
load = False

vitesse = 60

WIDTH, HEIGHT = 1000, 600

# Couleurs
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GRAY = (40, 40, 60)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
PURPLE = (255, 0, 255)

# Paramètres voiture
car_width, car_height = 40, 20
max_speed = 5
acceleration = 0.2
turn_speed = 4

add = 50
# Murs
walls = [
    pygame.Rect(50, 50, 700, 10),
    pygame.Rect(50, 540, 700, 10),
    pygame.Rect(50, 50, 10, 500),
    pygame.Rect(740, 50, 10, 500),
    pygame.Rect(200, 150, 400, 10),
    pygame.Rect(200, 150, 10, 300),
    pygame.Rect(400, 250, 10, 300),
    pygame.Rect(600, 150, 10, 300),
]

# Checkpoints
checkpoints = [
    (180, 110, 40),  # Départ/arrivée
    (670, 150, 40),
    (600, 495, 40),
    (400, 210, 40),
    (180, 485, 40)
]

# Actions possibles
ACTIONS = [
    (0, 0, 0),  # Ne rien faire
    (1, 0, 0),  # Accélérer
    (0, 1, 0),  # Freiner
    (0, 0, 1),  # Tourner droite
    (0, 0, -1),  # Tourner gauche
    (1, 0, 1),  # Accélérer + tourner droite
    (1, 0, -1),  # Accélérer + tourner gauche
    (0, 1, 1),  # Freiner + tourner droite
    (0, 1, -1),  # Freiner + tourner gauche
]




class Car:
    def __init__(self, agent=None):
        self.reset()
        self.agent = agent
        self.training = True

    def reset(self):
        self.x = 100
        self.y = 100
        self.angle = 0
        self.speed = 0
        self.alive = True
        self.checkpoints_reached = 0
        self.current_checkpoint = 0
        self.distance_traveled = 0
        self.last_checkpoint_distance = float('inf')
        self.stuck_timer = 0
        self.last_x, self.last_y = self.x, self.y
        self.start_time = pygame.time.get_ticks()
        self.last_state = None
        self.last_action = None
        self.step_count = 0
        self.total_reward = 0

    def get_state(self):
        # Capteurs de distance (5 directions)
        sensor_angles = [-45, -22.5, 0, 22.5, 45]
        distances = []

        for sensor_angle in sensor_angles:
            distance = self.cast_ray(self.angle + sensor_angle)
            distances.append(min(distance / 200.0, 1.0))  # Normaliser à [0,1]

        # Vitesse normalisée
        speed_norm = self.speed / max_speed

        # Distance et angle vers le prochain checkpoint
        cx, cy, _ = checkpoints[self.current_checkpoint]
        dx = cx - self.x
        dy = cy - self.y
        distance_to_checkpoint = math.sqrt(dx * dx + dy * dy) / 500.0  # Normaliser

        # Angle relatif vers le checkpoint
        target_angle = math.degrees(math.atan2(dy, dx))
        angle_diff = (target_angle - self.angle) % 360
        if angle_diff > 180:
            angle_diff -= 360
        angle_diff_norm = angle_diff / 180.0  # Normaliser à [-1,1]

        state = distances + [speed_norm, distance_to_checkpoint, angle_diff_norm]
        return np.array(state, dtype=np.float32)

    def cast_ray(self, angle):
        ray_x, ray_y = self.x, self.y
        step = 2
        max_distance = 200

        dx = math.cos(math.radians(angle)) * step
        dy = math.sin(math.radians(angle)) * step

        for i in range(int(max_distance / step)):
            ray_x += dx
            ray_y += dy

            if ray_x < 0 or ray_x >= WIDTH or ray_y < 0 or ray_y >= HEIGHT:
                return math.sqrt((ray_x - self.x) ** 2 + (ray_y - self.y) ** 2)

            for wall in walls:
                if wall.collidepoint(ray_x, ray_y):
                    return math.sqrt((ray_x - self.x) ** 2 + (ray_y - self.y) ** 2)

        return max_distance

    def calculate_reward(self, old_state, action):
        reward = 0

        # Récompense de base pour rester en vie
        reward += 1

        # Récompense pour se rapprocher du checkpoint
        cx, cy, _ = checkpoints[self.current_checkpoint]
        current_distance = math.sqrt((self.x - cx) ** 2 + (self.y - cy) ** 2)

        if current_distance < self.last_checkpoint_distance:
            reward += 5  # Se rapprocher du checkpoint
        else:
            reward -= 2  # S'éloigner du checkpoint

        self.last_checkpoint_distance = current_distance

        # Récompense pour la vitesse (encourager à aller vite)
        reward += self.speed * 2

        # Pénalité pour rester immobile
        if abs(self.speed) < 0.1:
            reward -= 5

        # Récompense énorme pour atteindre un checkpoint
        if hasattr(self, 'checkpoint_reached') and self.checkpoint_reached:
            reward += 100
            self.checkpoint_reached = False

        # Pénalité pour collision
        if not self.alive:
            reward -= 100

        return reward

    def update(self):
        if not self.alive:
            return

        current_state = self.get_state()

        # Action basée sur l'IA
        if self.agent:
            action_index = self.agent.act(current_state, self.training)
            accelerate, brake, turn = ACTIONS[action_index]
        else:
            # Action aléatoire si pas d'agent
            action_index = random.randint(0, len(ACTIONS) - 1)
            accelerate, brake, turn = ACTIONS[action_index]

        # Sauvegarder l'état et l'action précédents
        if self.last_state is not None and self.agent and self.training:
            reward = self.calculate_reward(self.last_state, self.last_action)
            self.total_reward += reward
            self.agent.remember(self.last_state, self.last_action, reward,
                                current_state, not self.alive)

        self.last_state = current_state.copy()
        self.last_action = action_index

        # Appliquer les actions
        old_x, old_y = self.x, self.y

        # Mise à jour de la vitesse
        if accelerate and not brake:
            self.speed = min(self.speed + acceleration, max_speed)
        elif brake:
            self.speed = max(self.speed - acceleration, -max_speed / 2)
        else:
            self.speed *= 0.95

        # Rotation
        if turn != 0:
            self.angle += turn * turn_speed

        # Mouvement
        dx = math.cos(math.radians(self.angle)) * self.speed
        dy = math.sin(math.radians(self.angle)) * self.speed
        new_x = self.x + dx
        new_y = self.y + dy

        # Vérifier les collisions
        if self.check_collision(new_x, new_y):
            self.alive = False
            if self.agent and self.training:
                reward = self.calculate_reward(current_state, action_index)
                self.agent.remember(current_state, action_index, reward,
                                    current_state, True)
            return

        self.x, self.y = new_x, new_y

        # Calculer la distance parcourue
        self.distance_traveled += math.sqrt((self.x - old_x) ** 2 + (self.y - old_y) ** 2)

        # Vérifier les checkpoints
        self.check_checkpoints()

        # Détecter si bloqué
        if abs(self.x - self.last_x) < 0.5 and abs(self.y - self.last_y) < 0.5:
            self.stuck_timer += 1
            if self.stuck_timer > 300:  # 5 secondes
                self.alive = False
        else:
            self.stuck_timer = 0
            self.last_x, self.last_y = self.x, self.y

        self.step_count += 1

        # Limite de temps par épisode
        if self.step_count > 3000:  # 50 secondes à 60 FPS
            self.alive = False

    def check_collision(self, x, y):
        car_rect = pygame.Rect(0, 0, car_width, car_height)
        car_rect.center = (x, y)
        for wall in walls:
            if car_rect.colliderect(wall):
                return True
        return False

    def check_checkpoints(self):
        cx, cy, radius = checkpoints[self.current_checkpoint]
        distance = math.hypot(self.x - cx, self.y - cy)

        if distance < radius:
            self.checkpoints_reached += 1
            self.current_checkpoint = (self.current_checkpoint + 1) % len(checkpoints)
            self.checkpoint_reached = True

            # Réinitialiser la distance au checkpoint
            if self.current_checkpoint < len(checkpoints):
                cx, cy, _ = checkpoints[self.current_checkpoint]
                self.last_checkpoint_distance = math.sqrt((self.x - cx) ** 2 + (self.y - cy) ** 2)

    def draw(self, screen, color=RED):
        if not self.alive:
            color = (100, 100, 100)

        car_surface = pygame.Surface((car_width, car_height))
        car_surface.fill(color)
        car_surface.set_colorkey(BLACK)
        rotated_car = pygame.transform.rotate(car_surface, -self.angle)
        rect = rotated_car.get_rect(center=(self.x, self.y))
        screen.blit(rotated_car, rect.topleft)

    def draw_sensors(self, screen):
        if not self.alive:
            return

        sensor_angles = [-45, -22.5, 0, 22.5, 45]
        for sensor_angle in sensor_angles:
            distance = self.cast_ray(self.angle + sensor_angle)
            end_x = self.x + math.cos(math.radians(self.angle + sensor_angle)) * distance
            end_y = self.y + math.sin(math.radians(self.angle + sensor_angle)) * distance
            pygame.draw.line(screen, BLUE, (self.x, self.y), (end_x, end_y), 1)


def draw_checkpoints(screen):
    font = pygame.font.Font(None, 24)
    for i, (x, y, radius) in enumerate(checkpoints):
        pygame.draw.circle(screen, BLUE, (x, y), radius, 3)
        label = "Départ" if i == 0 else f"{i}"
        text = font.render(label, True, WHITE)
        text_rect = text.get_rect(center=(x, y))
        screen.blit(text, text_rect)

def game_loop():
    if show:
        # Initialisation
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("IA de Course - DQN (Deep Q-Network)")

        clock = pygame.time.Clock()

    # Créer l'agent DQN
    agent = ia.DQNAgent()
    car = Car(agent)

    if load:
        agent.load(ia.save_filename)

    # Statistiques
    episode = 0
    episode_rewards = []
    best_checkpoints = 0
    show_sensors = True
    training_mode = True

    done = False

    if show:
        font = pygame.font.Font(None, 24)
        small_font = pygame.font.Font(None, 20)

    print("Début de l'entraînement DQN...")
    print("Epsilon initial:", agent.epsilon)

    while not done:
        if show:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_s:
                        show_sensors = not show_sensors
                    elif event.key == pygame.K_r:
                        car.reset()
                        episode += 1
                    elif event.key == pygame.K_t:
                        training_mode = not training_mode
                        car.training = training_mode
                        print(f"Mode {'entraînement' if training_mode else 'test'}")

        # Mise à jour de la voiture
        car.update()

        # Entraînement du réseau
        if training_mode and len(agent.memory) > agent.batch_size:
            agent.replay()

        # Nouvelle épisode si la voiture est morte
        if not car.alive:
            episode_rewards.append(car.total_reward)
            if car.checkpoints_reached > best_checkpoints:
                best_checkpoints = car.checkpoints_reached
                print(f"Nouveau record! Épisode {episode}: {best_checkpoints} checkpoints, "
                      f"Reward: {car.total_reward:.1f}, Epsilon: {agent.epsilon:.3f}")
                agent.save(ia.save_filename)


            car.reset()
            episode += 1


        if show:
            # Affichage
            screen.fill(GRAY)

            # Dessiner le circuit
            for wall in walls:
                pygame.draw.rect(screen, WHITE, wall)

            draw_checkpoints(screen)

            # Dessiner la voiture
            car.draw(screen, GREEN if car.alive else RED)
            if show_sensors and car.alive:
                car.draw_sensors(screen)

            # Interface utilisateur
            episode_text = font.render(f"Épisode: {episode}", True, WHITE)
            checkpoints_text = font.render(f"Checkpoints: {car.checkpoints_reached}", True, WHITE)
            best_text = font.render(f"Record: {best_checkpoints}", True, WHITE)
            reward_text = font.render(f"Reward: {car.total_reward:.1f}", True, WHITE)
            epsilon_text = font.render(f"Epsilon: {agent.epsilon:.3f}", True, WHITE)
            memory_text = font.render(f"Mémoire: {len(agent.memory)}", True, WHITE)
            mode_text = font.render(f"Mode: {'Entraînement' if training_mode else 'Test'}", True, WHITE)

        # Affichage des statistiques récentes
            if len(episode_rewards) > 0:
                recent_avg = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
                avg_text = font.render(f"Reward moy. (10): {recent_avg:.1f}", True, WHITE)
                screen.blit(avg_text, (10, 160))

            controls_text = small_font.render("S: Capteurs | R: Reset | T: Training/Test", True, WHITE)

            screen.blit(episode_text, (10, 10))
            screen.blit(checkpoints_text, (10, 35))
            screen.blit(best_text, (10, 60))
            screen.blit(reward_text, (10, 85))
            screen.blit(epsilon_text, (10, 110))
            screen.blit(memory_text, (10, 135))
            screen.blit(mode_text, (10, 185))
            screen.blit(controls_text, (10, HEIGHT - 25))

            pygame.display.flip()
            clock.tick(vitesse)
