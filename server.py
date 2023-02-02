from demo import p

from sanic import Sanic
from sanic.response import json
from path import Path

app = Sanic("sanic")

@app.route("/<input:str>")
async def main(request, input):
    result = p(input)
    return json(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8002)
