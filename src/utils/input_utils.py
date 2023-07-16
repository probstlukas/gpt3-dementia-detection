def get_user_input(prompt, choices):
    while True:
        user_input = input(prompt)
        if user_input.lower() in choices:
            return user_input.lower()
        else:
            print('Invalid input. Please try again.')