# Deploy Tensorflow model with Seldon Core
Model deployment on the Seldon Core could be applied with `SeldonDeployment` CRD. This resource defines component images, resources and inference graph for prediction service. In this case of the deployment of serving TensorFlow model, Seldon Core wraps TensorFlow Serving container, and pass the request from the user and return inferenced results which received from that container.

A request for the prediction could be fulfilled with REST or gRPC protocol.

## Example of the prediction request
### python
```
...

model_prediction_address = http://<ingress-address>/seldon/seldon/nwd-segmentation/api/v1.0/predictions

data = {
    'data': {
        'ndarray': images.numpy().tolist()
    }
}

res = requests.post(model_prediction_address,
                    headers={},
                    json=data,
                    verify=True,
                    cert=None)
```

### curl
`curl -H "Content-Type: application/json" --data '@data.json' -X POST http://<ingress-address>/seldon/seldon/nwd-segmentation/api/v1.0/predictions | jq '' > response.json`

**data.json**
```
{
  "data": {
    "ndarray": [
      [
        [
          [
            0.981735110282898,
            0.5251141786575317,
            0.8082191944122314
          ],
          [
            0.5159817934036255,
            -0.077625572681427,
            0.41552507877349854
          ],
          ...
          [
            0.9118943214416504,
            0.8061673641204834,
            0.8854625225067139
          ]
        ]
      ]
    ]
  }
}
```

**response.json**
```
{
  "data": {
    "names": [
      "t:0",
      "t:1",
      "t:2",
      ...
      "t:255"
    ],
    "ndarray": [
      [
        [
          [
            0.0265661236,
            -0.0845582
          ],
          [
            0.0857038796,
            -0.0818213671
          ],
          ...
          [
            0.014386232,
            0.000919547165
          ]
        ]
      ]
    ]
  },
  "meta": {
    "requestPath": {
      "segmentation": "seldonio/tfserving-proxy:1.8.0"
    }
  }
}
```

## References
* [inference graph](https://docs.seldon.io/projects/seldon-core/en/v1.8.0/graph/inference-graph.html)
