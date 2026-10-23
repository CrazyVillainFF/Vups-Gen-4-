from backend.skills import build_system_prompt
def test_prompt_names_assistant_and_coding_mode():
 p=build_system_prompt("coding"); assert "Vups Gen 4" in p and "senior software engineer" in p
def test_image_state_is_explicit():
 assert "An image was supplied" in build_system_prompt("vision",True)
 assert "No image was supplied" in build_system_prompt("vision",False)
