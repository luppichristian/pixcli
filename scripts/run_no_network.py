"""Run the CLI with all Python network connections and DNS lookups blocked.

Use only for verifying local image inference. It is not a system firewall and
must not be used for the animation backend, which needs a localhost server.
"""
import os
import socket


def blocked(*args, **kwargs):
    raise RuntimeError('Network access attempted during blocked-network verification')


if __name__ == '__main__':
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked
    socket.create_connection = blocked
    socket.getaddrinfo = blocked
    from pixcli.cli import app
    app()
