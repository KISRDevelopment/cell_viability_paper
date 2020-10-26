async function main()
{
    get_input([[0, 1], [0, 2]])

    // const model = await tf.loadLayersModel('models/yeast_gi_mn/model.json');

    // const example = tf.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.27740745585384186, 0.29908339124520045]]);

    // const prediction = model.predict(example);

    // prediction.print();
}

function get_input(id_pairs)
{
    fetch('./get_input', {
        headers: { "Content-Type": "application/json; charset=utf-8" },
        method: 'POST',
        body: JSON.stringify(id_pairs)
    })
    .then((r) => r.json())
    .then((F) => {
        console.log(F);
    })
}
main();
