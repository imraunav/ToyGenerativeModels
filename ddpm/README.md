> [!WARNING]
>
> Facing an issue with the cosine schedule, the samples blow up to neg or pos.
> Check this for more info: 
> - https://github.com/openai/guided-diffusion/issues/42

> Remark: Sample with `lower number of sampling steps` when using `cosine schedule` or slip beta values to 0.02 like original linear schedule.