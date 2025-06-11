# Import pygame library for graphics functionality
import pygame
# Define white color in RGB format (Red, Green, Blue)
WHITE = (255, 255, 255)

class Connection:
    def __init__(self, from_port, to_port):
        # Store the source and destination ports of the connection
        self.from_port = from_port
        self.to_port = to_port

        # Get the blocks that these ports belong to
        from_block = from_port.block
        to_block = to_port.block

        # Add the destination block to the source block's outputs if not already present
        if to_block not in from_block.outputs:
            from_block.outputs.append(to_block)
        # Add the source block to the destination block's inputs if not already present
        if from_block not in to_block.inputs:
            to_block.inputs.append(from_block)

        # Add this connection to both ports' connection lists
        from_port.connections.append(self)
        to_port.connections.append(self)

    def draw(self, surface):
        # Draw a white line between the two ports with a thickness of 3 pixels
        pygame.draw.line(surface, WHITE, self.from_port.position, self.to_port.position, 3)
