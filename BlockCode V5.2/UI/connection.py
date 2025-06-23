# Import pygame library for graphics functionality
import pygame
import math
# Define white color in RGB format (Red, Green, Blue)
WHITE = (255, 255, 255)
GLOW = (100, 200, 255)  # Brighter color for active connections

class Connection:
    def __init__(self, from_port, to_port):
        # Store the source and destination ports of the connection
        self.from_port = from_port
        self.to_port = to_port
        self.active = False  # Track if this connection is currently active
        self.glow_intensity = 0  # Track glow animation state

        # Get the blocks that these ports belong to
        from_block = from_port.block
        to_block = to_port.block

        # Add the destination block to the source block's outputs if not already present
        if to_block not in from_block.outputs:
            from_block.outputs.append(to_block)
            from_block.composite_block.output_composites.append(to_block.composite_block)
        # Add the source block to the destination block's inputs if not already present
        if from_block not in to_block.inputs:
            to_block.inputs.append(from_block)
            to_block.composite_block.input_composites.append(from_block.composite_block)

        # Add this connection to both ports' connection lists
        from_port.connections.append(self)
        to_port.connections.append(self)

    def draw(self, surface):
        # Draw the connection with glow effect if active
        if self.active:
            # Animate glow intensity
            self.glow_intensity = (self.glow_intensity + 0.1) % 1.0
            # Calculate glow color based on intensity
            glow_factor = abs(math.sin(self.glow_intensity * math.pi))
            glow_color = (
                int(GLOW[0] * glow_factor + WHITE[0] * (1 - glow_factor)),
                int(GLOW[1] * glow_factor + WHITE[1] * (1 - glow_factor)),
                int(GLOW[2] * glow_factor + WHITE[2] * (1 - glow_factor))
            )
            # Draw thicker line with glow color
            pygame.draw.line(surface, glow_color, self.from_port.position, self.to_port.position, 5)
        else:
            # Draw normal white line
            pygame.draw.line(surface, WHITE, self.from_port.position, self.to_port.position, 3)
