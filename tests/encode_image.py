import base64

import click


@click.command()
@click.option('--input', '-i', help='Input an image path')
def main(input):
    with open(input, 'rb') as image:
        click.echo(base64.b64encode(image.read()).decode())


if __name__ == '__main__':
    main()
