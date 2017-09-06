// $(document).ready(function(){
//     function createModel() {
//         alert('kuku');
//     }
//     $("#create_model").click(createModel);
//
//
// });

window.onload = function (){
    new Vue({
        el: '#app_control',
        data: {title: 'jjjdlkjf', models :[], mode: 'start', selected_model:'', work_header:''},
        // computed:{models: function(){$.get('/get_models', function(models){return models;})}},
        mounted: function() {
            this.get_models()
        },
        watch:{
            models:function(){},
            selected_model: function (value) {
                this.mode = 'model_details';
                this.work_header = 'Model Details';
                // this.selected_model = value;
                // alert(value);
            }
        },
        methods:{
            get_models: function () {
                var vm = this; // Keep reference to viewmodel object
                $.get('/get_models', function(data){
                    vm.models = JSON.parse(data);
                });
            },
            create_model: function () {
                this.mode = 'create_model';
                this.work_header = 'Create New Model';
            },
            delete_model: function () {
                alert('delete!!!')
            },
            train_model: function () {
                alert('train!!!')
            },
            predict_model: function () {
                alert('predict!!!')
            },
            post_create: function () {
                var model_name = this.$refs['model_name'].value;
                var img_rows = this.$refs['img_rows'].value;
                var img_cols = this.$refs['img_cols'].value;
                var epoch = this.$refs['epoch'].value;
                var kernel_size = this.$refs['kernel_size'].value;
                var pool_size = this.$refs['pool_size'].value;

                // TODO: add validation and show error msg.
                var vm = this;
                $.post('/add_model',{
                    'model_name': model_name,
                    'img_rows': img_rows,
                    'img_cols': img_cols,
                    'epoch': epoch,
                    'kernel_size': kernel_size,
                    'pool_size': pool_size,
                },function(data,status){
                    if(status=="success"){
                        models.append(model_name);
                        vm.$refs['model_name'].value = '';
                    }
                    else{
                        // TODO: show error msg
                    }
                })
            }

        }
    })
}