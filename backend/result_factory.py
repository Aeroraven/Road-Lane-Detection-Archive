import json


class ResultFactory:
    @staticmethod
    def build_success_result(data):
        return json.dumps({
            "code":200,
            "data":data
        })