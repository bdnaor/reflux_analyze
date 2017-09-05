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
        data: {title: 'jjjdlkjf', models :[]},
        // computed:{models: function(){$.get('/get_models', function(models){return models;})}},
        mounted: function() {
            this.get_models()
        },
        watch:{models:function(){}},
        methods:{
            get_models: function () {
                var vm = this; // Keep reference to viewmodel object
                $.get('/get_models', function(data){
                    vm.models = JSON.parse(data);
                });
            }
        }
    })
}