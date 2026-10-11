import importlib.util
import json
from pathlib import Path


def test_signed_stac_tree_excludes_unrelated_files(fixture_item, monkeypatch):
    path = Path(__file__).parents[1] / "scripts/stage-fixture.py"
    spec = importlib.util.spec_from_file_location("staging", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    objects = {}

    class Client:
        def generate_presigned_url(self, operation, Params, ExpiresIn):
            return "http://storage/" + Params["Bucket"] + "/" + Params["Key"] + "?signature=test"

        def put_object(self, Bucket, Key, Body, ContentType):
            objects[Key] = Body

    monkeypatch.setattr(module.boto3, "client", lambda *args, **kwargs: Client())
    (fixture_item.parent / "unrelated-private.json").write_text('{"secret": "preserve locally"}')
    url = module.publish(fixture_item.parent, "http://storage", "run", entrypoint="item.json")
    document = json.loads(objects["run/item.json"])
    assert url == "http://storage/results/run/item.json?signature=test"
    assert len(objects) == 13
    assert "run/unrelated-private.json" not in objects
    assert all("?signature=test" in asset["href"] for asset in document["assets"].values())
    assert "signature=test" not in fixture_item.read_text()
