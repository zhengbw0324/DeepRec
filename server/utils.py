



def parse_command_line_args(unparsed: list[str]) -> dict:

    args = {}
    for text_arg in unparsed:
        if '=' not in text_arg:
            # print(text_arg)
            raise ValueError(f"Invalid command line argument: {text_arg}, please add '=' to separate key and value.")
        key, value = text_arg.split('=')
        key = key[len('--'):]
        try:
            value = eval(value)
        except:
            pass
        args[key] = value
    return args