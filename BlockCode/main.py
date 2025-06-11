# Import required libraries and modules
import pygame
import sys
from block import Block
from connection import Connection
from model_block import LinearBlock, ReLUBlock, Conv2dBlock, CompositeBlock
from data_block import RandomTensorBlock, CSVtoTensorBlock
from export_running_code import run_and_save_running_code
from visualization import animate_data_flow
from operation_blocks import AddBlock, SumBlock, MatmulBlock
from run_block import InferenceBlock, TrainingBlock, EvaluationBlock

# Initialize pygame and create the main window
pygame.init()
screen = pygame.display.set_mode((1200, 800))
pygame.display.set_caption("Visual Graph Editor")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 16)
title_font = pygame.font.SysFont("Arial", 24)

# Global variables to store blocks and connections
blocks = []
connections = []
dragging_port = None
selected_blocks = set()  # Set to store selected blocks
showing_context_menu = False
context_menu_pos = (0, 0)
context_menu_options = ["Abstract Selected Blocks"]

# Define colors for the interface
DARK = (30, 30, 30)
LIGHT = (200, 200, 200)
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)
HIGHLIGHT = (100, 150, 255)

# Model creation UI state
showing_model_ui = False
selected_category = None
selected_model = None
model_name = ""

# Define categories and their options
categories = {
    "Layers": ["Linear Layer", "ReLU", "Conv2D"],
    "Data": ["RandomTensor", "CSVtoTensor"],
    "Operations": ["Add", "Sum", "Matmul"],
    "Run": ["Inference", "Training", "Evaluation"]
}

model_rects = []
category_rects = []
param_inputs = []  # List of TextInput objects for parameters
current_params = {}  # Dictionary to store current parameter values

# Run Code button
run_button_rect = pygame.Rect(20, 20, 120, 40)

class TextInput:
    def __init__(self, x, y, width, height, font, param_name="", param_type="str", default_value=""):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = str(default_value)
        self.font = font
        self.active = False
        self.color = GRAY
        self.text_color = WHITE
        self.cursor_visible = True
        self.cursor_timer = 0
        self.param_name = param_name
        self.param_type = param_type

    def get_value(self):
        if self.param_type == "int":
            try:
                return int(self.text)
            except ValueError:
                return 0
        elif self.param_type == "bool":
            return self.text.lower() in ("true", "1", "yes")
        elif self.param_type == "tuple":
            try:
                # Split by commas and convert each element to int
                return tuple(int(x.strip()) for x in self.text.split(','))
            except ValueError:
                return (64, 64)  # Default shape if parsing fails
        return self.text

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
            self.color = HIGHLIGHT if self.active else GRAY
            return False

        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_RETURN:
                return True
            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            else:
                if event.unicode.isprintable():
                    self.text += event.unicode
            return False
        return False

    def update(self):
        if self.active:
            self.cursor_timer = (self.cursor_timer + 1) % 60
            self.cursor_visible = self.cursor_timer < 30

    def draw(self, surface):
        # Draw the input box border
        pygame.draw.rect(surface, self.color, self.rect, 2)

        # Render the text
        text_surface = self.font.render(self.text, True, self.text_color)

        # Use font ascent/descent for better vertical alignment
        ascent = self.font.get_ascent()
        descent = self.font.get_descent()
        font_height = ascent + descent
        text_y = self.rect.y + (self.rect.height - font_height) // 2

        surface.blit(text_surface, (self.rect.x + 5, text_y))

        # Draw cursor if active and blinking
        if self.active and self.cursor_visible:
            cursor_x = self.rect.x + 5 + text_surface.get_width() + 2
            cursor_y_top = text_y
            cursor_y_bottom = text_y + font_height
            pygame.draw.line(surface, self.text_color,
                            (cursor_x, cursor_y_top),
                            (cursor_x, cursor_y_bottom), 2)




def get_port_under_mouse(pos):
    for block in blocks:
        for port in block.input_ports + block.output_ports:
            if (pygame.Vector2(port.position) - pos).length() < 8:
                return port
    return None

def dfs(block, visited=None):
    if visited is None:
        visited = set()
    if block in visited:
        return
    visited.add(block)
    print("Visited:", block.label, block.model_block.to_dict())
    for neighbor in block.outputs:
        dfs(neighbor, visited)


def draw_model_ui():
    # Create a larger surface for the UI
    s = pygame.Surface((500, 600))
    s.set_alpha(230)  # More opaque background
    s.fill(DARK)
    screen.blit(s, (350, 100))  # Centered position
    
    # Draw title
    title = title_font.render("Create New Block", True, WHITE)
    title_rect = title.get_rect(centerx=600, y=120)
    screen.blit(title, title_rect)
    
    # Calculate center position for content
    center_x = 600
    start_y = 170
    spacing = 50
    option_width = 300
    option_height = 40

    if not selected_category:
        # Draw category selection section
        type_label = font.render("Select Category:", True, WHITE)
        type_rect = type_label.get_rect(centerx=center_x, y=start_y)
        screen.blit(type_label, type_rect)

        y = start_y + 40
        category_rects.clear()
        for category in categories.keys():
            rect = pygame.Rect(center_x - option_width//2, y, option_width, option_height)
            category_rects.append((rect, category))
            color = HIGHLIGHT if category == selected_category else GRAY
            pygame.draw.rect(screen, color, rect, 2, border_radius=4)
            text = font.render(category, True, WHITE)
            text_rect = text.get_rect(centerx=rect.centerx, centery=rect.centery)
            screen.blit(text, text_rect)
            y += spacing

    elif not selected_model:
        # Draw model options for selected category
        type_label = font.render(f"Select {selected_category}:", True, WHITE)
        type_rect = type_label.get_rect(centerx=center_x, y=start_y)
        screen.blit(type_label, type_rect)

        y = start_y + 40
        model_rects.clear()
        for model in categories[selected_category]:
            rect = pygame.Rect(center_x - option_width//2, y, option_width, option_height)
            model_rects.append((rect, model))
            color = HIGHLIGHT if model == selected_model else GRAY
            pygame.draw.rect(screen, color, rect, 2, border_radius=4)
            text = font.render(model, True, WHITE)
            text_rect = text.get_rect(centerx=rect.centerx, centery=rect.centery)
            screen.blit(text, text_rect)
            y += spacing

        # Draw back button
        back_rect = pygame.Rect(center_x - 60, y + 20, 120, 30)
        pygame.draw.rect(screen, GRAY, back_rect, 2, border_radius=4)
        back_text = font.render("Back", True, WHITE)
        back_text_rect = back_text.get_rect(centerx=back_rect.centerx, centery=back_rect.centery)
        screen.blit(back_text, back_text_rect)
        model_rects.append((back_rect, "back"))
        
    else:
        # Draw parameter inputs section
        type_label = font.render(f"Configure {selected_model}:", True, WHITE)
        type_rect = type_label.get_rect(centerx=center_x, y=start_y)
        screen.blit(type_label, type_rect)

        y = start_y + 40
        for input_box in param_inputs:
            # Draw parameter name
            param_text = font.render(f"{input_box.param_name}:", True, WHITE)
            param_rect = param_text.get_rect(centerx=center_x, y=y)
            screen.blit(param_text, param_rect)
            
            # Update input box position and draw
            input_box.rect = pygame.Rect(center_x - option_width//2, y + 25, option_width, option_height)
            input_box.draw(screen)
            y += 80

        # Draw name input section
        name_label = font.render("Block Name:", True, WHITE)
        name_rect = name_label.get_rect(centerx=center_x, y=y)
        screen.blit(name_label, name_rect)
        
        name_input.rect = pygame.Rect(center_x - option_width//2, y + 25, option_width, option_height)
        name_input.draw(screen)
        y += 80

        # Draw back button
        back_rect = pygame.Rect(center_x - 60, y + 20, 120, 30)
        pygame.draw.rect(screen, GRAY, back_rect, 2, border_radius=4)
        back_text = font.render("Back", True, WHITE)
        back_text_rect = back_text.get_rect(centerx=back_rect.centerx, centery=back_rect.centery)
        screen.blit(back_text, back_text_rect)
        model_rects.append((back_rect, "back"))
        
        # Draw help text
        help_text = font.render("Press Enter to create, Esc to cancel", True, GRAY)
        help_rect = help_text.get_rect(centerx=center_x, y=y + 60)
        screen.blit(help_text, help_rect)

def handle_model_ui_events(event):
    global showing_model_ui, selected_category, selected_model, model_name, param_inputs, current_params

    if event.type == pygame.MOUSEBUTTONDOWN:
        if not showing_model_ui:
            return

        # Check if click is outside the UI
        if not pygame.Rect(350, 100, 500, 600).collidepoint(event.pos):
            showing_model_ui = False
            selected_category = None
            selected_model = None
            name_input.text = ""
            param_inputs.clear()
            current_params.clear()
            return

        # Handle category selection (first level)
        if not selected_category:
            for rect, category in category_rects:
                if rect.collidepoint(event.pos):
                    selected_category = category
                    selected_model = None
                    param_inputs.clear()
                    current_params.clear()
                    return

        # Handle model selection (second level)
        elif not selected_model:
            for rect, model in model_rects:
                if rect.collidepoint(event.pos):
                    if model == "back":
                        selected_category = None
                        param_inputs.clear()
                        current_params.clear()
                    else:
                        selected_model = model
                        # Create parameter inputs based on selected model
                        param_inputs.clear()
                        current_params.clear()
                        
                        # Get parameter info based on model type
                        if selected_model == "Linear Layer":
                            param_info = LinearBlock.get_param_info(None)
                        elif selected_model == "ReLU":
                            param_info = ReLUBlock.get_param_info(None)
                        elif selected_model == "Conv2D":
                            param_info = Conv2dBlock.get_param_info(None)
                        elif selected_model == "Add":
                            param_info = AddBlock.get_param_info(None)
                        elif selected_model == "Sum":
                            param_info = SumBlock.get_param_info(None)
                        elif selected_model == "Matmul":
                            param_info = MatmulBlock.get_param_info(None)
                        elif selected_model == "RandomTensor":
                            param_info = RandomTensorBlock.get_param_info(None)
                        elif selected_model == "CSVtoTensor":
                            param_info = CSVtoTensorBlock.get_param_info(None)
                        elif selected_model == "Inference":
                            param_info = InferenceBlock.get_param_info(None)
                        elif selected_model == "Training":
                            param_info = TrainingBlock.get_param_info(None)
                        elif selected_model == "Evaluation":
                            param_info = EvaluationBlock.get_param_info(None)
                        else:
                            param_info = []
                        
                        # Create input boxes for each parameter
                        y = 250
                        for param_name, param_type, default_value in param_info:
                            input_box = TextInput(380, y, 340, 40, font, param_name, param_type, default_value)
                            param_inputs.append(input_box)
                            y += 60
                    return

        # Handle parameter configuration (third level)
        else:
            # Check for back button first
            for rect, model in model_rects:
                if rect.collidepoint(event.pos) and model == "back":
                    selected_model = None
                    param_inputs.clear()
                    current_params.clear()
                    return

            # Handle parameter input clicks
            for input_box in param_inputs:
                if input_box.rect.collidepoint(event.pos):
                    input_box.active = True
                    input_box.color = HIGHLIGHT
                else:
                    input_box.active = False
                    input_box.color = GRAY

            if name_input.rect.collidepoint(event.pos):
                name_input.active = True
                name_input.color = HIGHLIGHT
            else:
                name_input.active = False
                name_input.color = GRAY

    if event.type == pygame.KEYDOWN and showing_model_ui:
        if event.key == pygame.K_ESCAPE:
            showing_model_ui = False
            selected_category = None
            selected_model = None
            name_input.text = ""
            param_inputs.clear()
            current_params.clear()
            return

        if selected_model:
            # Handle parameter input
            for input_box in param_inputs:
                if input_box.active:
                    if input_box.handle_event(event):
                        continue

            # Handle name input
            if name_input.active:
                if name_input.handle_event(event):
                    # Create the block when Enter is pressed
                    params_copy = {input_box.param_name: input_box.get_value() 
                                 for input_box in param_inputs}
                    create_block_from_type(100, 100, selected_model, name_input.text, params_copy)
                    showing_model_ui = False
                    selected_category = None
                    selected_model = None
                    name_input.text = ""
                    param_inputs.clear()
                    current_params.clear()
                    return

def create_block_from_type(x, y, block_type, name, params=None):
    run_block = None
    model_block = None
    data_block = None

    if params is None:
        params = {}
    
    # Get parameter info and set defaults for any missing parameters
    if block_type == "RandomTensor":
        param_info = RandomTensorBlock.get_param_info(None)
        data_block = RandomTensorBlock(name, params)
        num_inputs = data_block.get_num_input_ports()
        num_outputs = data_block.get_num_output_ports()
    elif block_type == "CSVtoTensor":
        param_info = CSVtoTensorBlock.get_param_info(None)
        data_block = CSVtoTensorBlock(name, params)
        num_inputs = data_block.get_num_input_ports()
        num_outputs = data_block.get_num_output_ports()
    elif block_type == "Inference":
        param_info = InferenceBlock.get_param_info(None)
        # Set default values for any missing parameters
        for param_name, _, default_value in param_info:
            if param_name not in params:
                params[param_name] = default_value
        run_block = InferenceBlock(name, params)
        num_inputs = run_block.get_num_input_ports()
        num_outputs = run_block.get_num_output_ports()
    elif block_type == "Training":
        param_info = TrainingBlock.get_param_info(None)
        # Set default values for any missing parameters
        for param_name, _, default_value in param_info:
            if param_name not in params:
                params[param_name] = default_value
        run_block = TrainingBlock(name, params)
        num_inputs = run_block.get_num_input_ports()
        num_outputs = run_block.get_num_output_ports()
    elif block_type == "Evaluation":
        param_info = EvaluationBlock.get_param_info(None)
        # Set default values for any missing parameters
        for param_name, _, default_value in param_info:
            if param_name not in params:
                params[param_name] = default_value
        run_block = EvaluationBlock(name, params)
        num_inputs = run_block.get_num_input_ports()
        num_outputs = run_block.get_num_output_ports()
    else:  # Handle ModelBlocks
        if block_type == "Linear Layer":
            param_info = LinearBlock.get_param_info(None)
            model_block = LinearBlock(name, params)
        elif block_type == "ReLU":
            param_info = ReLUBlock.get_param_info(None)
            model_block = ReLUBlock(name, params)
        elif block_type == "Conv2D":
            param_info = Conv2dBlock.get_param_info(None)
            model_block = Conv2dBlock(name, params)
        elif block_type == "Add":
            param_info = AddBlock.get_param_info(None)
            model_block = AddBlock(name, params)
        elif block_type == "Sum":
            param_info = SumBlock.get_param_info(None)
            model_block = SumBlock(name, params)
        elif block_type == "Matmul":
            param_info = MatmulBlock.get_param_info(None)
            model_block = MatmulBlock(name, params)
        else:
            raise ValueError("Unsupported block type")
        
        # Set default values for any missing parameters
        for param_name, _, default_value in param_info:
            if param_name not in params:
                params[param_name] = default_value
        
        num_inputs = model_block.get_num_input_ports()
        num_outputs = model_block.get_num_output_ports()

    # Create the block with all parameters
    blocks.append(Block(x, y, name, model_block, run_block, data_block, num_inputs, num_outputs))

name_input = TextInput(380, 250, 340, 40, font)

def handle_events():
    global dragging_port, showing_model_ui, selected_category, selected_model
    global showing_context_menu, context_menu_pos, selected_blocks
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit(); sys.exit()
            
        elif event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()
            
            # Handle right-click for context menu
            if event.button == 3:  # Right click
                if selected_blocks:
                    showing_context_menu = True
                    context_menu_pos = pos
                return
                
            # Handle left-click for selection
            elif event.button == 1:  # Left click
                # Handle run button click
                if run_button_rect.collidepoint(pos):
                    if blocks:
                        animate_data_flow(blocks, screen, font, clock, run_button_rect, connections)
                        # Run and save the model
                        run_and_save_running_code(blocks, "run_model")
                            
                    return
                
                if showing_model_ui:
                    handle_model_ui_events(event)
                else:
                    # Check if clicking on a block
                    block_clicked = False
                    for block in reversed(blocks):
                        if block.rect.collidepoint(pos):
                            block_clicked = True
                            if pygame.key.get_mods() & pygame.KMOD_META:  # Command key
                                if block in selected_blocks:
                                    selected_blocks.remove(block)
                                else:
                                    selected_blocks.add(block)
                            else:
                                selected_blocks.clear()
                                selected_blocks.add(block)
                                block.dragging = True
                                block.offset = (pos[0] - block.rect.x, pos[1] - block.rect.y)
                            break
                    
                    # If clicked empty space, clear selection
                    if not block_clicked and not (pygame.key.get_mods() & pygame.KMOD_META):
                        selected_blocks.clear()
                    
                    # Check for port connections
                    port = get_port_under_mouse(pygame.Vector2(pos))
                    if port and not port.is_input:
                        dragging_port = port

        elif event.type == pygame.MOUSEBUTTONUP:
            for block in blocks:
                block.dragging = False
            if dragging_port:
                target = get_port_under_mouse(pygame.Vector2(pygame.mouse.get_pos()))
                if target and target.is_input:
                    connections.append(Connection(dragging_port, target))
                dragging_port = None

        elif event.type == pygame.KEYDOWN:
            if showing_model_ui:
                handle_model_ui_events(event)
            elif event.key == pygame.K_c:
                showing_model_ui = True

def update_blocks():
    mx, my = pygame.mouse.get_pos()
    for block in blocks:
        if block.dragging:
            block.rect.topleft = (mx - block.offset[0], my - block.offset[1])
    if showing_model_ui:
        name_input.update()
        for input_box in param_inputs:
            input_box.update()

def draw():
    screen.fill(DARK)
    
    # Draw Run Code button
    pygame.draw.rect(screen, GRAY, run_button_rect, border_radius=6)
    run_text = font.render("Run Code", True, WHITE)
    screen.blit(run_text, (run_button_rect.x + 10, run_button_rect.y + 10))
    
    for conn in connections:
        conn.draw(screen)
    for block in blocks:
        # Draw selection highlight
        if block in selected_blocks:
            pygame.draw.rect(screen, HIGHLIGHT, block.rect, 2)
        block.draw(screen, font)
    if dragging_port:
        pygame.draw.line(screen, LIGHT, dragging_port.position, pygame.mouse.get_pos(), 2)
    if showing_model_ui:
        draw_model_ui()
    if showing_context_menu:
        draw_context_menu(screen)
    pygame.display.flip()

def draw_context_menu(surface):
    global showing_context_menu
    if not showing_context_menu:
        return
        
    menu_rect = pygame.Rect(context_menu_pos[0], context_menu_pos[1], 200, 30)
    pygame.draw.rect(surface, GRAY, menu_rect)
    text = font.render("Abstract Selected Blocks", True, WHITE)
    surface.blit(text, (menu_rect.x + 5, menu_rect.y + 5))
    
    # Check if mouse is over menu option
    mouse_pos = pygame.mouse.get_pos()
    if menu_rect.collidepoint(mouse_pos):
        pygame.draw.rect(surface, HIGHLIGHT, menu_rect, 2)
        
        # Handle click on menu option
        if pygame.mouse.get_pressed()[0]:  # Left click
            abstract_selected_blocks()
            showing_context_menu = False

def abstract_selected_blocks():
    print("\n=== Starting Block Abstraction ===")
    if len(selected_blocks) < 2:
        print("❌ Need at least 2 blocks to abstract!")
        return
        
    print(f"📦 Selected blocks: {[b.label for b in selected_blocks]}")
    
    # Create a new composite block
    composite_name = f"composite_{len(blocks)}"
    print(f"🏗️  Creating composite block: {composite_name}")
    
    # Get blocks in topological order
    ordered_blocks = []
    visited = set()
    temp_visited = set()
    
    def visit(block):
        if block in temp_visited:
            return
        if block in visited:
            return
        temp_visited.add(block)
        for input_block in block.inputs:
            if input_block in selected_blocks:
                visit(input_block)
        temp_visited.remove(block)
        visited.add(block)
        ordered_blocks.append(block)
    
    # Start with blocks that have no inputs
    for block in selected_blocks:
        if not block.inputs:
            visit(block)
    
    print(f"📋 Ordered blocks: {[b.label for b in ordered_blocks]}")
    
    # Create composite with ordered blocks
    composite = CompositeBlock(composite_name, [block.model_block for block in ordered_blocks])
    
    # Find the input and output blocks
    input_blocks = [b for b in selected_blocks if not b.inputs]
    output_blocks = [b for b in selected_blocks if not b.outputs]
    print(f"📥 Input blocks: {[b.label for b in input_blocks]}")
    print(f"📤 Output blocks: {[b.label for b in output_blocks]}")
    
    # Calculate the center position of all selected blocks
    x_sum = sum(block.rect.centerx for block in selected_blocks)
    y_sum = sum(block.rect.centery for block in selected_blocks)
    center_x = x_sum / len(selected_blocks)
    center_y = y_sum / len(selected_blocks)
    print(f"📍 New block position: ({center_x}, {center_y})")
    
    # Create a new block with the composite at the center position
    new_block = Block(center_x - 60, center_y - 40, composite_name, composite, 
                     num_inputs=len(input_blocks), 
                     num_outputs=len(output_blocks))
    
    # Store connections to reconnect later
    incoming_connections = []
    outgoing_connections = []
    for block in selected_blocks:
        # Store incoming connections
        for input_block in block.inputs:
            if input_block not in selected_blocks:
                for conn in connections:
                    if conn.to_port.block == block and conn.from_port.block == input_block:
                        incoming_connections.append((input_block, conn.from_port))
        
        # Store outgoing connections
        for output_block in block.outputs:
            if output_block not in selected_blocks:
                for conn in connections:
                    if conn.from_port.block == block and conn.to_port.block == output_block:
                        outgoing_connections.append((output_block, conn.to_port))
    
    print(f"🔌 Found {len(incoming_connections)} incoming and {len(outgoing_connections)} outgoing connections")
    
    print("🗑️  Removing old blocks...")
    # Remove old blocks and their connections
    for block in selected_blocks:
        if block in blocks:
            blocks.remove(block)
            print(f"  - Removed: {block.label}")
        # Remove connections involving this block
        connections[:] = [conn for conn in connections 
                        if conn.from_port.block != block and conn.to_port.block != block]
    
    # Add new block
    blocks.append(new_block)
    print(f"✨ Added new composite block: {new_block.label}")
    
    # Reconnect inputs
    for i, (input_block, from_port) in enumerate(incoming_connections):
        if i < len(new_block.input_ports):
            connections.append(Connection(from_port, new_block.input_ports[i]))
            print(f"  - Reconnected input {i} from {input_block.label}")
    
    # Reconnect outputs
    for i, (output_block, to_port) in enumerate(outgoing_connections):
        if i < len(new_block.output_ports):
            connections.append(Connection(new_block.output_ports[i], to_port))
            print(f"  - Reconnected output {i} to {output_block.label}")
    
    # Clear selection
    selected_blocks.clear()
    showing_context_menu = False
    print("✅ Abstraction complete!\n")

while True:
    handle_events()
    update_blocks()
    draw()
    clock.tick(60)
