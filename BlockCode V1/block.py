# Import pygame for graphics functionality
import pygame
# Import Port class for creating input/output ports
from port import Port

# Define color constants in RGB format
GRAY = (60, 60, 60)      # Background color for blocks
WHITE = (255, 255, 255)  # Text color
BLUE = (100, 150, 255)   # Output port color
GREEN = (100, 255, 150)  # Input port color

class Block:
    def __init__(self, x, y, label, model_block=None, run_block=None, data_block=None, data_converter=None, num_inputs=1, num_outputs=1):
        # Initialize block properties
        self.label = label
        # Create a rectangle for the block's visual representation
        self.rect = pygame.Rect(x, y, 120, 80)
        # Track if block is being dragged
        self.dragging = False
        # Store mouse offset for dragging
        self.offset = (0, 0)
        # Lists to store input and output ports
        self.input_ports = []
        self.output_ports = []

        # Store the model, run, and data blocks
        self.model_block = model_block
        self.run_block = run_block
        self.data_block = data_block
        self.data_converter = data_converter
        
        # Lists to store connected blocks
        self.inputs = []
        self.outputs = []
        # Create the specified number of input and output ports
        self._create_ports(num_inputs, num_outputs)

    def _create_ports(self, num_inputs, num_outputs):
        # Calculate spacing between ports based on block height
        spacing = self.rect.height // (max(num_inputs, num_outputs) + 1)
        # Create input ports on the left side
        for i in range(num_inputs):
            self.input_ports.append(Port(self, (0, spacing * (i + 1)), is_input=True))
        # Create output ports on the right side
        for i in range(num_outputs):
            self.output_ports.append(Port(self, (self.rect.width, spacing * (i + 1)), is_input=False))

    def draw(self, surface, font):
        # Draw the block's background rectangle
        pygame.draw.rect(surface, GRAY, self.rect, border_radius=6)
        # Render and draw the block's label
        text = font.render(self.label, True, WHITE)
        surface.blit(text, (self.rect.x + 10, self.rect.y + 10))
        # Draw all ports with appropriate colors
        for port in self.input_ports + self.output_ports:
            color = GREEN if port.is_input else BLUE
            pygame.draw.circle(surface, color, port.position, 8)

    def __hash__(self):
        # Use object's memory address as hash value
        return id(self)

    def __eq__(self, other):
        # Compare blocks by their memory address
        return id(self) == id(other)
    

class LogicBlock(Block):
    def get_block_role(self) -> str:
        """
        What is this block's functional identity?
        Used for graph validation and role assignment.
        Examples: "model", "data_source", "preprocessor", "runner", "metric"
        """
        return "generic"

    def get_input_type(self) -> str:
        """
        What type of input does this block expect?
        Examples: "Tensor", "TokenDict", "Str", "Model"
        """
        return "Any"  # Or "None" for data sources

    def get_output_type(self) -> str:
        """
        What type of data does this block produce?
        Examples: "Tensor", "TokenDict", "Str", "Model"
        """
        return "Unknown"

