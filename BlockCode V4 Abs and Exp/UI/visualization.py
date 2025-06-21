import pygame
from typing import List, Set
from UI.block import Block

def get_block_layers(blocks: List[Block]) -> List[List[Block]]:
    """Get blocks organized in layers using BFS."""
    # Find input blocks (blocks with no inputs)
    input_blocks = [block for block in blocks if not block.inputs]
    if not input_blocks:
        return []
    
    # Initialize layers with input blocks
    layers = [input_blocks]
    visited = set(input_blocks)
    
    # BFS to find subsequent layers
    while True:
        current_layer = layers[-1]
        next_layer = []
        
        # Find all blocks that can be processed in the next layer
        for block in current_layer:
            for output_block in block.outputs:
                # Check if all inputs of this block are in previous layers
                if output_block not in visited and all(input_block in visited for input_block in output_block.inputs):
                    next_layer.append(output_block)
                    visited.add(output_block)
        
        if not next_layer:
            break
            
        layers.append(next_layer)
    
    return layers

def animate_data_flow(blocks: List[Block], screen, font, clock, run_button_rect, connections):
    """Animate the data flow through the network layer by layer."""
    print("Starting animation...")
    
    # Get layers for animation
    layers = get_block_layers(blocks)
    print(f"Found {len(layers)} layers")
    
    # Animate each layer
    for i, layer in enumerate(layers):
        print(f"Animating layer {i+1} with {len(layer)} blocks")
        
        # Activate connections for this layer
        for block in layer:
            # Activate all connections from this block
            for port in block.output_ports:
                for conn in port.connections:
                    conn.active = True
                    print(f"Activated connection from {block.label}")
        
        # Draw the current state
        screen.fill((30, 30, 30))  # DARK color
        
        # Draw Run Code button
        pygame.draw.rect(screen, (100, 100, 100), run_button_rect, border_radius=6)  # GRAY color
        run_text = font.render("Run Code", True, (255, 255, 255))  # WHITE color
        screen.blit(run_text, (run_button_rect.x + 10, run_button_rect.y + 10))
        
        # Draw connections and blocks
        for conn in connections:
            conn.draw(screen)
        for block in blocks:
            block.draw(screen, font)
        
        pygame.display.flip()
        
        # Wait for animation
        pygame.time.wait(1000)  # Increased to 1000ms (1 second) for better visibility
        
        # Deactivate connections for this layer
        for block in layer:
            for port in block.output_ports:
                for conn in port.connections:
                    conn.active = False
                    print(f"Deactivated connection from {block.label}")
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    # Reset all connections
                    for conn in connections:
                        conn.active = False
                    return False
        
        clock.tick(60)
    
    print("Animation complete!")
    return True 